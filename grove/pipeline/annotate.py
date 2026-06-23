"""Annotate: draw detector boxes onto image copies for human QC (CLAUDE.md §6.6).

These previews exist for ONE reason: so a human can SANITY-CHECK that the
auto-labels landed on the right fruit before trusting them (CLAUDE.md §1, §6.6).

Internalize the project's central reframe (§1): the deliverable of Grove is the
LABEL DATA (the coordinate files written by export.py), NOT these pictures. The
annotated previews here are throwaway visual aids — never the training output. If
a preview and the exported coordinates ever disagree, the coordinates are right by
definition and the preview is the thing to fix. So this stage is deliberately
cheap and optional: it reads the current boxes, paints them, and writes JPEGs/PNGs
to ``<work_dir>/previews/`` for eyeballing. Nothing downstream reads its output.

Import discipline: this is a GPU-SIDE stage (it runs in the Kaggle notebook
alongside detect). cv2 + supervision are heavy and are NOT installed on the local
[review] Mac, so importing this module there will fail — that is expected and
fine, because annotate never runs locally. Keeping cv2/supervision at module level
(rather than lazy) is therefore acceptable here.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

from grove.core.config import Config
from grove.core.models import ImageRecord, Manifest

logger = logging.getLogger(__name__)


def annotate(cfg: Config) -> None:
    """Draw current boxes onto image copies in ``<work_dir>/previews/``.

    Source manifest priority (we want to preview the most recent detector
    output, but still produce something useful right after ingest):
        1. predictions.json  — boxes filled by `grove detect` (the usual case)
        2. manifest.json     — ingest-only (no boxes yet; previews are just copies)

    Unreadable images are skipped and logged rather than aborting the run
    (mirrors ingest/detect resilience) — one bad photo must not kill QC for the
    rest of the folder. Returns None; the side effect is the previews on disk.
    """
    work_dir = Path(cfg.paths.work_dir)

    # Resolve the source of boxes. predictions.json (post-detect) is preferred;
    # fall back to manifest.json so `grove annotate` is still runnable right after
    # ingest (it just draws nothing, confirming images load + paths resolve).
    predictions_path = work_dir / "predictions.json"
    manifest_path = work_dir / "manifest.json"
    if predictions_path.exists():
        source = predictions_path
    elif manifest_path.exists():
        source = manifest_path
    else:
        raise FileNotFoundError(
            f"No manifest to annotate in {work_dir} "
            f"(looked for predictions.json, manifest.json). "
            f"Run `grove ingest` (and `grove detect`) first."
        )

    manifest = Manifest.load(source)
    logger.info("Annotating previews from %s", source)

    previews_dir = work_dir / "previews"
    previews_dir.mkdir(parents=True, exist_ok=True)

    # supervision annotators are stateless and reusable across images, so build
    # them once. BoxAnnotator draws the rectangles; LabelAnnotator writes the
    # "<label> <score>" text. These are purely cosmetic — see module docstring.
    #
    # color_lookup=ColorLookup.INDEX is REQUIRED here: both annotators default to
    # ColorLookup.CLASS, which resolves each box's color from detections.class_id.
    # Our Detections (built below) carry no class_id — this is a single-class
    # preview that draws straight from canonical boxes — so the CLASS default would
    # raise "Could not resolve color by class ..." on every image with >=1 box,
    # silently producing previews with no boxes drawn. INDEX colors by position
    # and needs no class_id.
    box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
    label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)

    written = 0
    skipped = 0
    for record in manifest.images:
        try:
            _annotate_one(record, work_dir, previews_dir, box_annotator, label_annotator)
            written += 1
        except Exception as exc:
            # Cheap & optional stage: never let a single bad/unreadable image
            # abort QC for the whole folder.
            skipped += 1
            logger.warning("skipping preview for %s: %s", record.id, exc)

    logger.info(
        "annotate complete: %d preview(s) written, %d skipped -> %s",
        written,
        skipped,
        previews_dir,
    )


def _annotate_one(
    record: ImageRecord,
    work_dir: Path,
    previews_dir: Path,
    box_annotator: "sv.BoxAnnotator",
    label_annotator: "sv.LabelAnnotator",
) -> None:
    """Read one prepared image, paint its detections, write the preview.

    record.path is RELATIVE to work_dir (shared layout), so the real file is
    ``work_dir / record.path``. We read BGR with cv2 (supervision's expected
    color order) and draw into a copy.
    """
    image_path = work_dir / record.path
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        # cv2 returns None (not an exception) for unreadable/corrupt files.
        raise OSError(f"cv2 could not read image: {image_path}")

    H, W = image_bgr.shape[:2]

    # Build a supervision Detections from the record's CANONICAL boxes. Canonical
    # is normalized xyxy in [0,1]; supervision wants ABSOLUTE PIXEL xyxy, so we
    # convert each box via box.to_pixel_xyxy(W, H) (one place owns this math; see
    # CLAUDE.md §9). Coordinate bugs here are silent and look "almost right", so
    # we trust the canonical converters rather than hand-rolling the scaling.
    if record.detections:
        xyxy = np.array(
            [det.box.to_pixel_xyxy(W, H) for det in record.detections],
            dtype=np.float32,
        )
        # Per-box label text: "<class> <score>" when a score exists (detector
        # output), just "<class>" otherwise (e.g. human-drawn boxes have none).
        labels = [
            f"{det.label} {det.score:.2f}" if det.score is not None else det.label
            for det in record.detections
        ]
    else:
        # No boxes: supervision needs an empty (0, 4) array, not an empty list.
        xyxy = np.empty((0, 4), dtype=np.float32)
        labels = []

    detections = sv.Detections(xyxy=xyxy)

    # Draw into a copy so we never mutate the source pixels on disk.
    annotated = box_annotator.annotate(scene=image_bgr.copy(), detections=detections)
    annotated = label_annotator.annotate(
        scene=annotated, detections=detections, labels=labels
    )

    # Write the preview under the same filename (id + original extension) so it is
    # easy to correlate a preview with its source image.
    out_path = previews_dir / Path(record.path).name
    if not cv2.imwrite(str(out_path), annotated):
        raise OSError(f"cv2 could not write preview: {out_path}")
