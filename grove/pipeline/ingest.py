"""Ingest: scan an image folder -> EXIF-normalized images + Manifest (SPEC.md §6.3).

This is the first pipeline stage and the producer of the data contract every
later stage reads. It does three jobs:

1. Recursively find images under ``cfg.paths.input_dir`` (.jpg/.jpeg/.png/.webp).
2. For each image, normalize EXIF orientation so the *pixels* are upright and the
   stored orientation tag is dropped. This matters because boxes are recorded in
   coordinates relative to the upright pixel grid; if the orientation tag stayed,
   the detector's view, the reviewer's view, and the stored boxes could all
   disagree (SPEC.md §6.3, §12 EXIF pitfall). We fix it ONCE here so nothing
   downstream ever has to think about EXIF again.
3. Write the prepared images to a FLAT ``<work_dir>/images/<id>.<ext>`` directory
   and emit ``<work_dir>/manifest.json`` (images + run provenance, detections
   empty). ``ImageRecord.path`` is stored RELATIVE to work_dir ("images/<id>.<ext>")
   so the dataset stays portable between the Kaggle GPU machine and the local Mac.

Import discipline: this module is GPU-free — Pillow + std lib + grove.core only.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

import grove
from grove.core.config import Config
from grove.core.models import ImageRecord, Manifest

logger = logging.getLogger(__name__)

# Accepted image suffixes (compared case-insensitively against the lowercased
# file extension). Matches SPEC.md §6.3.
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

# Characters we allow to survive verbatim in an id slug. ids are used both as
# REST path segments and as on-disk filenames, so they must be URL- and
# filename-safe — in particular NO slashes (see the shared id rules).
_SAFE_ID_CHARS = set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
)


def ingest(cfg: Config) -> Manifest:
    """Scan ``cfg.paths.input_dir`` and build (and persist) a Manifest.

    Returns the in-memory Manifest and writes ``<work_dir>/manifest.json``.
    Unreadable/corrupt files are skipped and logged rather than crashing the run
    (SPEC.md §6.3) — one bad photo in a folder of hundreds must not abort the
    whole labeling pass.
    """
    cfg.paths.ensure_dirs()  # auto-create work_dir + export_dir (§10.4)

    input_dir = Path(cfg.paths.input_dir)
    work_dir = Path(cfg.paths.work_dir)
    images_out_dir = work_dir / "images"
    images_out_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.is_dir():
        # A missing input dir is a real operator error worth surfacing loudly,
        # not silently masking with an empty manifest.
        raise FileNotFoundError(f"input_dir does not exist or is not a directory: {input_dir}")

    # Deterministic file order so ids/collision-suffixes are stable across runs.
    candidates = sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
    )

    records: list[ImageRecord] = []
    used_ids: set[str] = set()
    found = 0
    skipped = 0

    for src in candidates:
        found += 1
        try:
            rel = src.relative_to(input_dir)
            image_id = _make_unique_id(rel, used_ids)
            out_name, out_path = _prepare_image(src, image_id, images_out_dir)

            # Record path RELATIVE to work_dir so the manifest is portable across
            # machines (Kaggle GPU -> local Mac). Every stage re-resolves it as
            # Path(cfg.paths.work_dir) / record.path.
            rel_path = f"images/{out_name}"
            with Image.open(out_path) as prepared:
                width, height = prepared.size

            records.append(
                ImageRecord(
                    id=image_id,
                    path=rel_path,
                    width=width,
                    height=height,
                    detections=[],
                    status="pending",
                )
            )
            used_ids.add(image_id)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            # Corrupt/truncated/unreadable file: skip it, keep the run alive.
            skipped += 1
            logger.warning("skipping unreadable image %s: %s", src, exc)

    manifest = Manifest.new(
        records,
        detector_backend=cfg.detector.backend,
        ontology=cfg.detector.ontology,
        box_threshold=cfg.detector.box_threshold,
        text_threshold=cfg.detector.text_threshold,
        # Record the full resolved tiling settings for provenance (§6.1).
        tiling=cfg.tiling.model_dump(),
        grove_version=grove.__version__,
    )

    manifest_path = work_dir / "manifest.json"
    manifest.save(manifest_path)

    logger.info(
        "ingest complete: %d image(s) found, %d ingested, %d skipped -> %s",
        found,
        len(records),
        skipped,
        manifest_path,
    )
    return manifest


def _prepare_image(src: Path, image_id: str, images_out_dir: Path) -> tuple[str, Path]:
    """Write an EXIF-upright copy of ``src`` to ``images_out_dir/<id>.<ext>``.

    Returns (output_filename, output_path).

    Behavior (SPEC.md §6.3):
    - Apply EXIF orientation via ImageOps.exif_transpose so the saved pixels are
      upright, then ensure the saved file carries NO orientation tag.
    - If exif_transpose actually rotated/flipped the pixels, re-encode the
      transposed image. If orientation was already normal, copy the original
      bytes verbatim — no needless re-encode (avoids JPEG generation loss).
    - Use the original extension, normalizing ``.jpeg`` -> ``.jpg`` for the
      output filename so downstream code sees a single canonical jpg suffix.
    """
    out_suffix = src.suffix.lower()
    if out_suffix == ".jpeg":
        out_suffix = ".jpg"
    out_name = f"{image_id}{out_suffix}"
    out_path = images_out_dir / out_name

    with Image.open(src) as img:
        # exif_transpose returns a NEW image with pixels rotated to match the
        # EXIF orientation, and (per Pillow) without the orientation tag in its
        # info — exactly what we want stored on disk.
        transposed = ImageOps.exif_transpose(img)

        # Detect whether orientation actually changed. If the pixel dimensions
        # differ, a 90/270-degree rotation happened; for 180/flip the size is
        # unchanged, so we also check for an orientation EXIF tag (272..274 area;
        # the orientation tag id is 0x0112). When either says "rotation applied"
        # OR there was an orientation tag, we re-encode the upright pixels;
        # otherwise we copy the bytes verbatim.
        size_changed = transposed.size != img.size
        had_orientation = _has_orientation_tag(img)

        if size_changed or had_orientation:
            # Re-encode the upright pixels. We do NOT pass through the original
            # EXIF, which guarantees no orientation tag survives.
            save_kwargs: dict = {}
            fmt = img.format  # original format hint (e.g. "JPEG", "PNG", "WEBP")
            to_save = transposed
            if out_suffix in (".jpg",):
                # JPEG cannot hold an alpha channel; flatten if present.
                if to_save.mode in ("RGBA", "P", "LA"):
                    to_save = to_save.convert("RGB")
                save_kwargs["quality"] = 95
                fmt = "JPEG"
            to_save.save(out_path, format=fmt, **save_kwargs)
        else:
            # Orientation already normal and no tag to strip: copy verbatim to
            # preserve original quality/bytes.
            shutil.copyfile(src, out_path)

    return out_name, out_path


def _has_orientation_tag(img: Image.Image) -> bool:
    """True if the image carries an EXIF Orientation tag (id 0x0112)."""
    try:
        exif = img.getexif()
    except Exception:  # pragma: no cover - defensive; some formats lack EXIF
        return False
    # 0x0112 (274) is the EXIF Orientation tag. A value of 1 means "normal",
    # but if the tag is present at all we re-encode to guarantee it is dropped.
    return 0x0112 in exif


def _make_unique_id(rel_path: Path, used_ids: set[str]) -> str:
    """Derive a stable, URL- & filename-safe id slug from a path relative to
    input_dir, ensuring uniqueness against ``used_ids``.

    Rule (shared id rules): drop the extension, replace os.sep and whitespace
    with "_", keep only [a-zA-Z0-9._-] (other chars -> "_"). On collision append
    "_2", "_3", ... so two distinct source paths never share an id (ids double as
    REST path segments and filenames, so they must be unique and slash-free).
    """
    # Use the POSIX form then strip the suffix so the slug is OS-independent and
    # never contains a path separator.
    stem_parts = rel_path.with_suffix("").parts
    raw = "_".join(stem_parts)

    chars: list[str] = []
    for ch in raw:
        if ch in _SAFE_ID_CHARS:
            chars.append(ch)
        else:
            # Includes whitespace and any leftover separators -> "_".
            chars.append("_")
    slug = "".join(chars)

    if not slug:
        # Pathological: a name made entirely of stripped chars. Fall back to a
        # generic stem so we still produce a valid, unique id.
        slug = "image"

    candidate = slug
    n = 2
    while candidate in used_ids:
        candidate = f"{slug}_{n}"
        n += 1
    return candidate
