"""Detect: run an open-vocab detector over the manifest -> predictions (CLAUDE.md §6.4, §7).

This is the GPU-hungry stage. It loads ``<work_dir>/manifest.json`` (produced by
ingest), constructs the configured detector (optionally wrapped in the tiling
detector for small/dense fruit), runs it on every image, and writes the filled
predictions out to ``<work_dir>/predictions.json``.

TEACHER, NOT THE ROBOT'S MODEL (CLAUDE.md §2/§12). Grounding DINO here is the slow,
accurate "teacher" used to *label* data — it is never the robot's deployable
runtime. Only the distilled YOLO student (pipeline/train.py) is robot-deployable.
The boxes written here are *drafts*: the mandatory human review step (review/)
turns them into a trustworthy dataset.

RESUMABILITY (CLAUDE.md §7, §10 — the central design requirement of this stage).
A dropped Kaggle session must not waste the scarce weekly GPU quota by re-detecting
images already done. So:
  * If ``<work_dir>/predictions.json`` already exists, we load it and carry forward
    every image that was already processed in that prior run, re-running ONLY the
    images still missing results.
  * We persist predictions.json INCREMENTALLY (every N images and once at the end),
    so a crash mid-run preserves all progress up to the last checkpoint.

"Already processed" is keyed by image id present in the prior predictions, NOT by
"has >=1 detection" — an image can legitimately finish with zero detections (no
fruit found), and we must not redo such images forever. The prior predictions
file is the record of what has been *run*.

Import discipline: this is GPU-side, so importing cv2 + numpy at module top is fine.
The detector itself is built lazily via get_detector (heavy backend imported there).
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2  # GPU-side stage: OpenCV at module top is acceptable here.

import grove
from grove.core.config import Config
from grove.core.models import ImageRecord, Manifest
from grove.pipeline.detectors.base import Detector, get_detector
from grove.pipeline.tiling import TiledDetector

logger = logging.getLogger(__name__)

# How often (in newly-detected images) to flush predictions.json to disk. Small
# enough that a crash loses little work; large enough to avoid thrashing the disk
# on every single image. Also flushed once at the very end regardless.
_CHECKPOINT_EVERY = 10


def detect(cfg: Config) -> Manifest:
    """Run the configured detector over the manifest and persist predictions.

    Loads ``<work_dir>/manifest.json``, fills each ImageRecord's detections with
    CANONICAL boxes from the detector, and writes ``<work_dir>/predictions.json``.
    Resumable and incrementally saved (see module docstring).

    Returns the in-memory predictions Manifest.
    """
    work_dir = Path(cfg.paths.work_dir)
    manifest_path = work_dir / "manifest.json"
    predictions_path = work_dir / "predictions.json"

    if not manifest_path.is_file():
        # Detect cannot run before ingest — surface this as a clear operator error
        # rather than producing an empty/confusing predictions file.
        raise FileNotFoundError(
            f"manifest not found at {manifest_path}; run `grove ingest` first"
        )

    manifest = Manifest.load(manifest_path)

    # --- Resume: pull forward results from any prior predictions run ----------
    # done_ids = images already PROCESSED in a previous (possibly interrupted)
    # detect run. We index their detections by id and skip re-detecting them.
    prior_detections: dict[str, list] = {}
    if predictions_path.is_file():
        try:
            prior = Manifest.load(predictions_path)
            # Only carry forward ids that still exist in the current manifest, so a
            # changed input folder can't resurrect stale records.
            current_ids = {img.id for img in manifest.images}
            for img in prior.images:
                if img.id in current_ids:
                    prior_detections[img.id] = img.detections
            logger.info(
                "resuming: %d image(s) already detected in %s will be skipped",
                len(prior_detections),
                predictions_path,
            )
        except (OSError, ValueError) as exc:
            # A corrupt/partial predictions.json must not block a fresh run; warn
            # and start over rather than crashing the whole stage.
            logger.warning(
                "could not load prior predictions %s (%s); starting fresh",
                predictions_path,
                exc,
            )

    # --- Build the (possibly tiled) detector ---------------------------------
    # get_detector lazily imports the heavy backend module. Wrap it in the tiling
    # detector when enabled so distant/small fruit isn't lost at full downscale
    # (§6.5). TiledDetector honors the same Detector contract, so detection code
    # below is identical either way.
    base_detector: Detector = get_detector(cfg)
    if cfg.tiling.enabled:
        detector: Detector = TiledDetector(
            base=base_detector,
            tile_size=cfg.tiling.tile_size,
            overlap=cfg.tiling.overlap,
            nms_iou=cfg.tiling.nms_iou,
        )
        logger.info(
            "tiling ENABLED (tile_size=%d, overlap=%.2f, nms_iou=%.2f)",
            cfg.tiling.tile_size,
            cfg.tiling.overlap,
            cfg.tiling.nms_iou,
        )
    else:
        detector = base_detector

    # --- Detection loop ------------------------------------------------------
    total = len(manifest.images)
    result_records: list[ImageRecord] = []
    newly_detected = 0  # counts images detected THIS run (for checkpoint cadence)
    skipped_resumed = 0
    skipped_unreadable = 0

    for i, record in enumerate(manifest.images, start=1):
        # Resume: if this image was already processed, carry its results forward
        # untouched and don't burn GPU time on it again (§7/§10).
        if record.id in prior_detections:
            record.detections = prior_detections[record.id]
            result_records.append(record)
            skipped_resumed += 1
            logger.info("[%d/%d] %s: resumed (already detected)", i, total, record.id)
            continue

        # Resolve the prepared image path: ImageRecord.path is RELATIVE to work_dir
        # (the portable contract), so the on-disk path is work_dir / record.path.
        image_path = work_dir / record.path
        # cv2.imread returns an HxWx3 uint8 BGR ndarray — exactly the Detector
        # contract's expected input. Returns None on failure (missing/corrupt).
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            # Skip & log rather than crash (mirrors ingest's posture); one bad file
            # must not abort the whole labeling pass. Carry the record forward with
            # whatever detections it has (none) so it still appears in predictions.
            skipped_unreadable += 1
            logger.warning(
                "[%d/%d] %s: could not read image %s; skipping",
                i,
                total,
                record.id,
                image_path,
            )
            result_records.append(record)
            continue

        # Run the detector. Output boxes are already CANONICAL (the backend/tiling
        # layer guarantees normalized xyxy in [0,1]); assign straight to the record.
        detections = detector.detect(image_bgr)
        record.detections = detections
        result_records.append(record)
        newly_detected += 1
        logger.info(
            "[%d/%d] %s: %d detection(s)", i, total, record.id, len(detections)
        )

        # Incremental checkpoint: persist progress periodically so a crash loses at
        # most ~_CHECKPOINT_EVERY images of GPU work, not the whole session.
        if newly_detected % _CHECKPOINT_EVERY == 0:
            _save_predictions(
                cfg, manifest, result_records, predictions_path
            )
            logger.info(
                "checkpoint: saved %d/%d image(s) to %s",
                len(result_records),
                total,
                predictions_path,
            )

    # --- Finalize ------------------------------------------------------------
    predictions = _save_predictions(cfg, manifest, result_records, predictions_path)
    logger.info(
        "detect complete: %d image(s) total — %d newly detected, %d resumed, "
        "%d unreadable -> %s",
        total,
        newly_detected,
        skipped_resumed,
        skipped_unreadable,
        predictions_path,
    )
    return predictions


def _save_predictions(
    cfg: Config,
    source_manifest: Manifest,
    records: list[ImageRecord],
    predictions_path: Path,
) -> Manifest:
    """Build a predictions Manifest from ``records`` and write it to
    ``predictions_path``.

    Detector/tiling provenance (backend, ontology, thresholds, tiling) is taken
    from the LIVE ``cfg`` used by THIS detect run — not from the loaded ingest
    manifest — because detect() drives its actual behavior from cfg (the detector
    is built from cfg.detector and tiling is gated on cfg.tiling). The config can
    legitimately change between `grove ingest` and `grove detect` (threshold/prompt/
    tiling sweeps, §8/§11), so copying the ingest manifest's settings would silently
    misrecord how these predictions were produced. Recording cfg keeps the file
    traceable to the exact run settings (§6.1, §12 non-determinism pitfall).
    grove_version is refreshed to the version actually doing the detecting;
    model_version/created_at are non-detector provenance carried from the manifest.
    """
    predictions = Manifest.new(
        records,
        detector_backend=cfg.detector.backend,
        ontology=cfg.detector.ontology,
        box_threshold=cfg.detector.box_threshold,
        text_threshold=cfg.detector.text_threshold,
        tiling=cfg.tiling.model_dump(),
        grove_version=grove.__version__,
        model_version=source_manifest.model_version,
        created_at=source_manifest.created_at,
    )
    predictions.save(predictions_path)
    return predictions
