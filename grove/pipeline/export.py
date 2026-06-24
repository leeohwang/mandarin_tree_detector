"""Dataset export — write the REAL deliverable (SPEC.md §1, §6.7, §16).

This module turns the current annotation state into a standard object-detection
dataset in YOLO and/or COCO format. It is deliberately *format-agnostic about its
source*: it reads whatever Manifest it is given (raw predictions pre-review, or
human corrections post-review) and writes labels. The exporter "doesn't care
which" (§6.7) — it just serializes the boxes it is handed.

Remember the project's central reframe (§1): the coordinate files written here are
the deliverable; the annotated previews are not. So correctness and format of the
output matter above all, and we VALIDATE on the way out (§6.7, §16): every label
references a real class id and no box falls outside the image.

This file is LIGHT (SPEC.md core import discipline): std lib + pyyaml + grove
core only — NO cv2 / PIL / supervision / ultralytics. It runs both on the GPU side
(`grove export`) and locally during review re-export, where the heavy extras are
not installed. Image pixel dimensions come from the Manifest (recorded at ingest);
image *files* are copied byte-for-byte with shutil, so we never need an image lib.
"""

from __future__ import annotations

import json
import logging
import random
import shutil
from pathlib import Path

import yaml

from grove.core.config import Config
from grove.core.models import Manifest

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# The shared writer. Both the CLI wrapper below and the review server's
# POST /api/export call into this so there is exactly one place that knows the
# on-disk dataset layout and the validation rules.
# -----------------------------------------------------------------------------
def write_dataset(
    manifest: Manifest,
    classes: list[str],
    work_dir: str,
    export_dir: str,
    formats: list[str],
    val_split: float,
    seed: int,
) -> dict:
    """Serialize ``manifest``'s current boxes into a YOLO + COCO dataset on disk.

    Args:
        manifest: the source of truth for boxes (predictions OR corrections).
        classes: canonical class list; the INDEX into this list is the class id
            written into YOLO labels and COCO categories. Order is load-bearing
            and must stay stable across runs (SPEC.md §6.2).
        work_dir: where the prepared images live (record.path is relative to it).
        export_dir: dataset output root (cleaned + recreated for idempotency).
        formats: subset of {"yolo", "coco"} controlling which formats are written.
        val_split: fraction of images placed in the val split (rounded).
        seed: RNG seed for the split so two identical runs match exactly (§12).

    Returns:
        A summary dict: {"train", "val", "boxes", "classes", "export_dir"}.

    Raises:
        ValueError: on an unknown class label, a class list with no usable name,
            or any box that violates the [0,1] / image-bounds invariants on write.
    """
    work_root = Path(work_dir)
    export_root = Path(export_dir)

    # Class name -> stable id (index). Built once; reused for YOLO + COCO so the
    # two formats can never disagree about which integer means "mandarin".
    name_to_id = {name: idx for idx, name in enumerate(classes)}

    # --- choose which images go into the dataset ----------------------------
    # Skipped images are explicitly excluded by the reviewer; everything else
    # (pending or reviewed) is included. Crucially, images with ZERO boxes are
    # STILL included: an empty YOLO label file is a valid "negative" example and
    # teaches the student that some frames legitimately contain no fruit.
    included = [img for img in manifest.images if img.status != "skipped"]

    # --- deterministic train/val split --------------------------------------
    # Sort by id first so the input order can't affect the split, then shuffle
    # with a seeded RNG. Same seed + same ids => identical split, every run (§12).
    ids_sorted = sorted(img.id for img in included)
    rng = random.Random(seed)
    shuffled = list(ids_sorted)
    rng.shuffle(shuffled)
    n_val = round(len(shuffled) * val_split)
    val_ids = set(shuffled[:n_val])  # first n_val after shuffle become val

    by_id = {img.id: img for img in included}

    # COCO requires integer image ids; derive a stable 1-based numbering from the
    # sorted id list once (position + 1) so train/val share consistent ids and
    # re-exports are identical. Precomputed to keep the per-image loop O(n).
    coco_int_id = {iid: i + 1 for i, iid in enumerate(ids_sorted)}

    want_yolo = "yolo" in formats
    want_coco = "coco" in formats

    # --- (re)create a clean output tree -------------------------------------
    # Wipe the whole export_dir so a re-export never leaves stale labels/images
    # from a previous run behind (idempotent re-export, §6.7).
    if export_root.exists():
        shutil.rmtree(export_root)
    images_dirs = {
        "train": export_root / "images" / "train",
        "val": export_root / "images" / "val",
    }
    labels_dirs = {
        "train": export_root / "labels" / "train",
        "val": export_root / "labels" / "val",
    }
    annotations_dir = export_root / "annotations"
    for d in (*images_dirs.values(), *labels_dirs.values(), annotations_dir):
        d.mkdir(parents=True, exist_ok=True)

    # COCO accumulators, per split. COCO uses 1-based monotonic ids by
    # convention; we generate them as we go.
    coco_images = {"train": [], "val": []}
    coco_annotations = {"train": [], "val": []}
    coco_ann_id = {"train": 1, "val": 1}

    counts = {"train": 0, "val": 0}
    total_boxes = 0

    for image_id in ids_sorted:
        record = by_id[image_id]
        split = "val" if image_id in val_ids else "train"
        counts[split] += 1

        W, H = record.width, record.height
        ext = Path(record.path).suffix  # preserve original extension (.jpg/.png/...)
        file_name = f"{record.id}{ext}"

        # Copy the prepared image (record.path is relative to work_dir, §shared
        # layout). copy2 preserves mtime; we never decode the image here.
        src = work_root / record.path
        dst_img = images_dirs[split] / file_name
        shutil.copy2(src, dst_img)

        # --- YOLO label file (one .txt per image, even if empty) ------------
        yolo_lines: list[str] = []
        # COCO image entry is needed whenever COCO is requested, regardless of
        # whether this image has any boxes.
        coco_image_entry = {
            "id": coco_int_id[image_id],
            "file_name": file_name,
            "width": W,
            "height": H,
        }

        for det in record.detections:
            label = det.label
            if label not in name_to_id:
                raise ValueError(
                    f"image {record.id!r}: detection label {label!r} is not in the "
                    f"configured class list {classes!r}; cannot assign a stable class id"
                )
            class_id = name_to_id[label]
            total_boxes += 1

            if want_yolo:
                cx, cy, w, h = det.box.to_yolo()
                # Validate on the way out: YOLO values are normalized in [0,1].
                # The canonical BBox already clamps, but we assert here so a bad
                # value can never silently corrupt the training labels (§6.7/§16).
                for name, val in (("cx", cx), ("cy", cy), ("w", w), ("h", h)):
                    if not 0.0 <= val <= 1.0:
                        raise ValueError(
                            f"image {record.id!r}: YOLO {name}={val} out of [0,1] "
                            f"for box {det.box!r}"
                        )
                yolo_lines.append(
                    f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                )

            if want_coco:
                x_min, y_min, bw, bh = det.box.to_coco(W, H)
                # Validate bounds in absolute pixels: the box must sit fully
                # inside the image. A small epsilon absorbs float rounding at the
                # far edge (e.g. x2==1.0 -> x_min+bw == W exactly).
                eps = 1e-6
                if (
                    x_min < -eps
                    or y_min < -eps
                    or x_min + bw > W + eps
                    or y_min + bh > H + eps
                ):
                    raise ValueError(
                        f"image {record.id!r}: COCO bbox [{x_min},{y_min},{bw},{bh}] "
                        f"exceeds image bounds {W}x{H}"
                    )
                coco_annotations[split].append(
                    {
                        "id": coco_ann_id[split],
                        "image_id": coco_image_entry["id"],
                        "category_id": class_id,
                        "bbox": [x_min, y_min, bw, bh],
                        "area": bw * bh,
                        "iscrowd": 0,
                    }
                )
                coco_ann_id[split] += 1

        if want_yolo:
            # Empty file when there are no boxes => valid YOLO negative example.
            (labels_dirs[split] / f"{record.id}.txt").write_text(
                "\n".join(yolo_lines) + ("\n" if yolo_lines else ""),
                encoding="utf-8",
            )

        if want_coco:
            coco_images[split].append(coco_image_entry)

    # --- data.yaml (YOLO) ---------------------------------------------------
    # Always written when YOLO is requested: ultralytics reads class names + nc
    # and the train/val image dirs from here.
    if want_yolo:
        data_yaml = {
            "path": str(export_root.resolve()),  # absolute root for portability
            "train": "images/train",
            "val": "images/val",
            "nc": len(classes),
            "names": list(classes),
        }
        (export_root / "data.yaml").write_text(
            yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8"
        )

    # --- COCO instances_{train,val}.json ------------------------------------
    if want_coco:
        # category_id matches the YOLO class id (index into classes) so the two
        # formats describe the SAME label space.
        categories = [
            {"id": idx, "name": name} for idx, name in enumerate(classes)
        ]
        for split in ("train", "val"):
            coco_doc = {
                "images": coco_images[split],
                "annotations": coco_annotations[split],
                "categories": categories,
            }
            (annotations_dir / f"instances_{split}.json").write_text(
                json.dumps(coco_doc, indent=2), encoding="utf-8"
            )

    summary = {
        "train": counts["train"],
        "val": counts["val"],
        "boxes": total_boxes,
        "classes": list(classes),
        "export_dir": str(export_root.resolve()),
    }
    logger.info(
        "Exported dataset: %d train / %d val images, %d boxes, formats=%s -> %s",
        summary["train"],
        summary["val"],
        summary["boxes"],
        sorted(f for f in ("yolo", "coco") if f in formats),
        summary["export_dir"],
    )
    return summary


# -----------------------------------------------------------------------------
# CLI wrapper: resolve the "current state" source, then delegate to write_dataset.
# -----------------------------------------------------------------------------
def export(cfg: Config) -> dict:
    """Export the dataset from the current annotation state (CLI entrypoint).

    Source priority (most-corrected first, SPEC.md §6.7 / shared layout):
        1. review_store.json  — human corrections (the reviewed truth)
        2. predictions.json   — raw auto-labels (pre-review)
        3. manifest.json      — ingest-only (boxes empty; negatives-only dataset)

    The first of these that exists and parses as a Manifest is used. We try the
    next on a parse failure so a malformed/partial store can't block export of
    the underlying predictions.
    """
    work_dir = Path(cfg.paths.work_dir)
    candidates = [
        work_dir / "review_store.json",
        work_dir / "predictions.json",
        work_dir / "manifest.json",
    ]

    manifest: Manifest | None = None
    used: Path | None = None
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            manifest = Manifest.load(candidate)
            used = candidate
            break
        except Exception as exc:  # malformed/non-Manifest JSON -> fall back
            logger.warning(
                "Could not load %s as a Manifest (%s); trying next source",
                candidate,
                exc,
            )

    if manifest is None:
        raise FileNotFoundError(
            f"No usable annotation state found in {work_dir} "
            f"(looked for review_store.json, predictions.json, manifest.json). "
            f"Run `grove ingest` (and `grove detect`) first."
        )

    logger.info("Exporting from %s", used)
    return write_dataset(
        manifest,
        classes=cfg.classes,
        work_dir=cfg.paths.work_dir,
        export_dir=cfg.paths.export_dir,
        formats=cfg.export.formats,
        val_split=cfg.export.val_split,
        seed=cfg.export.seed,
    )
