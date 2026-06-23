"""Annotation store: the source of truth for human corrections (CLAUDE.md §6.8).

This is the LOCAL side's authoritative state. The review UI seeds it from the
detector's raw predictions on first load, then every box edit / status change a
reviewer makes is persisted here. The store is what `export` re-reads to produce
the final, human-corrected dataset.

Why a separate file (CLAUDE.md §6.8, §12 — "corrections must NOT destroy raw
predictions"): we keep <work_dir>/review_store.json strictly distinct from
<work_dir>/predictions.json. The predictions file is the RAW auto-label output
and is never touched after detect writes it — so we can always measure how much
the teacher (Grounding DINO) got wrong by diffing the two. The store is a deep
copy that the human is free to mutate.

Import discipline: LOCAL/light side only. std lib + grove.core. No FastAPI here
(the HTTP layer lives in server.py); no GPU/heavy deps. server.py wraps these
methods in endpoints, so the method shapes mirror the REST contract.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from grove.core.config import Config
from grove.core.models import BBox, Detection, ImageRecord, Manifest

logger = logging.getLogger(__name__)

# File names within work_dir. predictions.json is preferred as the seed source
# because it carries detections; manifest.json is the fallback (detections
# empty) so the reviewer can still label a folder that was never auto-labeled.
_STORE_NAME = "review_store.json"
_PREDICTIONS_NAME = "predictions.json"
_MANIFEST_NAME = "manifest.json"

_VALID_STATUSES = {"pending", "reviewed", "skipped"}


class AnnotationStore:
    """In-memory working Manifest backed by <work_dir>/review_store.json.

    On construction the store either resumes an existing review session (load the
    store file as-is) or starts a fresh one by deep-copying the seed Manifest
    (predictions preferred, else manifest) and immediately persisting it. The raw
    predictions file is NEVER modified — corrections only ever flow into the
    separate store file.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._work_dir = Path(cfg.paths.work_dir)
        self._store_path = self._work_dir / _STORE_NAME

        if self._store_path.exists():
            # Resume: trust the store as the current corrected state verbatim.
            self._manifest = Manifest.load(self._store_path)
            logger.info("Resumed review store from %s", self._store_path)
        else:
            # First load: seed from the raw predictions (or manifest) WITHOUT
            # mutating the source. A deep copy guarantees that subsequent edits
            # can never reach back into the read-only predictions Manifest.
            seed = self._load_seed()
            self._manifest = seed.model_copy(deep=True)
            self.save()
            logger.info(
                "Seeded review store at %s from %s",
                self._store_path,
                self._seed_source,
            )

        # id -> ImageRecord index over the in-memory manifest. The records are
        # the SAME objects held in self._manifest.images, so mutating them in
        # place keeps the index and the manifest consistent.
        self._index: dict[str, ImageRecord] = self._manifest.image_by_id()

    # --- seeding -------------------------------------------------------------
    def _load_seed(self) -> Manifest:
        """Locate and load the seed Manifest: predictions.json then manifest.json.

        Records which file we used (self._seed_source) for logging. Raises a
        clear error if neither exists — there is nothing to review otherwise.
        """
        predictions = self._work_dir / _PREDICTIONS_NAME
        manifest = self._work_dir / _MANIFEST_NAME
        if predictions.exists():
            self._seed_source = predictions
            return Manifest.load(predictions)
        if manifest.exists():
            self._seed_source = manifest
            return Manifest.load(manifest)
        raise FileNotFoundError(
            f"No seed for review store: expected {predictions} or {manifest}. "
            "Run `grove detect` (or at least `grove ingest`) first."
        )

    # --- progress / listing --------------------------------------------------
    def _progress(self) -> dict[str, int]:
        """Counts by review status, for the queue/progress UI."""
        reviewed = skipped = pending = 0
        for img in self._manifest.images:
            if img.status == "reviewed":
                reviewed += 1
            elif img.status == "skipped":
                skipped += 1
            else:
                pending += 1
        return {
            "total": len(self._manifest.images),
            "reviewed": reviewed,
            "skipped": skipped,
            "pending": pending,
        }

    def meta(self) -> dict:
        """Project-level metadata for GET /api/meta."""
        return {
            "project_name": self.cfg.project_name,
            "classes": self.cfg.classes,
            "progress": self._progress(),
        }

    def list_images(self) -> dict:
        """The review queue: id + status + box count per image, plus progress."""
        images = [
            {
                "id": img.id,
                "status": img.status,
                "box_count": len(img.detections),
            }
            for img in self._manifest.images
        ]
        return {"images": images, "progress": self._progress()}

    # --- single image --------------------------------------------------------
    def get_image(self, image_id: str) -> dict | None:
        """Full state for one image (boxes in CANONICAL coords), or None if the
        id is unknown (so the server can answer 404)."""
        rec = self._index.get(image_id)
        if rec is None:
            return None
        boxes = [
            {
                "label": det.label,
                "score": det.score,
                "x1": det.box.x1,
                "y1": det.box.y1,
                "x2": det.box.x2,
                "y2": det.box.y2,
            }
            for det in rec.detections
        ]
        return {
            "id": rec.id,
            "width": rec.width,
            "height": rec.height,
            "status": rec.status,
            "boxes": boxes,
        }

    def image_path(self, image_id: str) -> Path | None:
        """Absolute path to the prepared image on disk, or None if id unknown.

        ImageRecord.path is RELATIVE TO work_dir ("images/<id>.<ext>"), so the
        real location is always work_dir / record.path (CLAUDE.md shared layout).
        """
        rec = self._index.get(image_id)
        if rec is None:
            return None
        return (self._work_dir / rec.path).resolve()

    # --- mutations -----------------------------------------------------------
    def set_boxes(self, image_id: str, boxes: list[dict]) -> bool:
        """Replace this image's detections with the corrected set, then persist.

        boxes is a list of {label, x1, y1, x2, y2, score?} in CANONICAL coords.
        Each box is funnelled through BBox, which clamps to [0,1] and RAISES on a
        degenerate box; we drop invalid boxes rather than fail the whole save, so
        a single bad box from the UI can't block the reviewer's progress.
        Returns False if the id is unknown (404-able).
        """
        rec = self._index.get(image_id)
        if rec is None:
            return False

        detections: list[Detection] = []
        for raw in boxes:
            try:
                box = BBox(
                    x1=raw["x1"],
                    y1=raw["y1"],
                    x2=raw["x2"],
                    y2=raw["y2"],
                )
            except (ValueError, KeyError, TypeError) as exc:
                # Degenerate/malformed box: skip it but keep going.
                logger.warning("Dropping invalid box for %s: %s", image_id, exc)
                continue
            detections.append(
                Detection(
                    label=str(raw.get("label", "")),
                    box=box,
                    score=raw.get("score"),
                )
            )

        rec.detections = detections
        self.save()
        return True

    def set_status(self, image_id: str, status: str) -> bool:
        """Set the review status (pending/reviewed/skipped) and persist.

        Returns False if the id is unknown; raises ValueError on a bad status so
        the server can answer 400 rather than silently accepting garbage.
        """
        if status not in _VALID_STATUSES:
            raise ValueError(
                f"invalid status {status!r}; must be one of {sorted(_VALID_STATUSES)}"
            )
        rec = self._index.get(image_id)
        if rec is None:
            return False
        rec.status = status  # type: ignore[assignment]  # validated above
        self.save()
        return True

    # --- export / persistence ------------------------------------------------
    def to_manifest(self) -> Manifest:
        """Return the current corrected working state for re-export.

        This is the live in-memory Manifest (carrying the human's corrections
        plus the original provenance), which export.write_dataset consumes.
        """
        return self._manifest

    def save(self) -> None:
        """Persist the working Manifest atomically (temp file + os.replace) so a
        crash mid-write can never leave a half-written, unparseable store and
        lose a reviewer's corrections."""
        self._work_dir.mkdir(parents=True, exist_ok=True)
        payload = self._manifest.model_dump_json(indent=2)
        # Write to a temp file in the SAME directory (os.replace is only atomic
        # within a filesystem), then atomically swap it into place.
        fd, tmp_name = tempfile.mkstemp(
            dir=self._work_dir, prefix=".review_store.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
            os.replace(tmp_name, self._store_path)
        except BaseException:
            # Clean up the temp file on any failure so we don't litter work_dir.
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
