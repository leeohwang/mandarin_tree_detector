"""Ingest tests: dimensions, EXIF normalization, id rules, skip-corrupt (CLAUDE.md §6.3, §13).

§6.3 makes ingest the producer of the data contract everything downstream reads,
so the things that can silently break the whole pipeline get pinned here:

  * pixel dimensions are read correctly;
  * EXIF orientation is NORMALIZED at ingest — the prepared file's pixels are
    upright (post-rotation width/height) AND it carries no orientation tag, so the
    detector's view, the reviewer's view, and the stored boxes can never disagree
    (§12 EXIF pitfall);
  * ids are stable, unique, and slash-free (they double as REST path segments and
    filenames);
  * a corrupt/garbage file is SKIPPED, not fatal (one bad photo must not abort a
    folder of hundreds);
  * prepared images land at work_dir/images/<id>.<ext> with record.path relative;
  * manifest.json round-trips via Manifest.load.

Pillow only (installed); NO cv2/torch. Config is built in code via temp_config.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps

import grove
from grove.core.models import Manifest
from grove.pipeline.ingest import ingest


def _write_jpeg(path: Path, width: int, height: int, color=(30, 60, 90), orientation: int | None = None):
    """Write a JPEG, optionally embedding an EXIF orientation tag (0x0112)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (width, height), color)
    if orientation is not None:
        exif = img.getexif()
        exif[0x0112] = orientation
        img.save(path, format="JPEG", exif=exif)
    else:
        img.save(path, format="JPEG")


def test_ingest_reads_dimensions_and_writes_manifest(temp_config):
    """Plain images: dimensions read; manifest written and round-trips."""
    in_dir = Path(temp_config.paths.input_dir)
    _write_jpeg(in_dir / "a.jpg", width=800, height=600)  # landscape
    # A second, portrait, PNG to exercise a non-jpg suffix + non-square shape.
    Image.new("RGB", (400, 700), (10, 10, 10)).save(in_dir / "b.png")

    manifest = ingest(temp_config)

    by_id = manifest.image_by_id()
    # Both images ingested, with their (non-square) dimensions read correctly.
    rec_a = manifest.get("a")
    assert rec_a is not None
    assert (rec_a.width, rec_a.height) == (800, 600)
    rec_b = manifest.get("b")
    assert rec_b is not None
    assert (rec_b.width, rec_b.height) == (400, 700)

    # Provenance recorded (§6.1).
    assert manifest.grove_version == grove.__version__
    assert manifest.detector_backend == temp_config.detector.backend

    # manifest.json exists on disk and round-trips.
    manifest_path = Path(temp_config.paths.work_dir) / "manifest.json"
    assert manifest_path.is_file()
    reloaded = Manifest.load(manifest_path)
    assert {i.id for i in reloaded.images} == set(by_id)


def test_ingest_normalizes_exif_orientation(temp_config):
    """An orientation=6 (90deg) JPEG -> prepared file is rotated upright + tag dropped.

    Stored size is 120x80 with orientation=6; after normalization the upright
    pixels are 80x120, and the prepared file must carry NO orientation tag.
    """
    in_dir = Path(temp_config.paths.input_dir)
    _write_jpeg(in_dir / "rotated.jpg", width=120, height=80, orientation=6)

    manifest = ingest(temp_config)
    rec = manifest.get("rotated")
    assert rec is not None

    # The MANIFEST dimensions reflect the upright pixels (90deg rotation swaps W/H).
    assert (rec.width, rec.height) == (80, 120)

    # The prepared file on disk matches and has no orientation tag left.
    prepared = Path(temp_config.paths.work_dir) / rec.path
    assert prepared.is_file()
    with Image.open(prepared) as out:
        assert out.size == (80, 120)
        assert 0x0112 not in out.getexif(), "orientation tag must be dropped"
        # Defensive: exif_transpose on the prepared file is now a no-op (already upright).
        assert ImageOps.exif_transpose(out).size == out.size


def test_ingest_ids_stable_unique_and_slash_free(temp_config):
    """ids derive from the relative path, are slash-free, and collisions get suffixed."""
    in_dir = Path(temp_config.paths.input_dir)
    # Two files in different subdirs whose slugged stems would COLLIDE: a nested
    # "row1/tree.jpg" slugs to "row1_tree"; we add a top-level "row1_tree.jpg" to
    # force a collision and assert the suffixing rule.
    _write_jpeg(in_dir / "row1" / "tree.jpg", 100, 80)
    _write_jpeg(in_dir / "row1_tree.jpg", 100, 80)

    manifest = ingest(temp_config)
    ids = [i.id for i in manifest.images]

    # No slashes anywhere (ids are REST path segments + filenames).
    for i in ids:
        assert "/" not in i and "\\" not in i
    # Unique.
    assert len(ids) == len(set(ids))
    # The slug + the collision suffix are both present.
    assert "row1_tree" in ids
    assert "row1_tree_2" in ids

    # Stability: a second ingest over the same input yields the SAME ids.
    manifest2 = ingest(temp_config)
    assert sorted(i.id for i in manifest2.images) == sorted(ids)


def test_ingest_skips_corrupt_file(temp_config):
    """A garbage 'image' is skipped (logged, not fatal); good images survive."""
    in_dir = Path(temp_config.paths.input_dir)
    _write_jpeg(in_dir / "good.jpg", 200, 150)
    # Not a real image: bytes with an image extension.
    (in_dir / "broken.jpg").write_bytes(b"this is definitely not a JPEG")

    manifest = ingest(temp_config)  # must NOT raise
    ids = {i.id for i in manifest.images}
    assert "good" in ids
    assert "broken" not in ids  # the corrupt file was skipped


def test_ingest_prepared_path_is_relative_and_resolves(temp_config):
    """record.path is 'images/<id>.<ext>' relative to work_dir and resolves to a file."""
    in_dir = Path(temp_config.paths.input_dir)
    _write_jpeg(in_dir / "x.jpg", 320, 240)

    manifest = ingest(temp_config)
    rec = manifest.get("x")
    assert rec is not None
    assert rec.path == "images/x.jpg"  # FLAT images dir, portable relative path
    resolved = Path(temp_config.paths.work_dir) / rec.path
    assert resolved.is_file()
