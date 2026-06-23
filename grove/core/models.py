"""Canonical data model for Grove (CLAUDE.md §6.1).

These Pydantic v2 models are the data contract shared across the GPU side
(pipeline) and the local side (review UI). They communicate ONLY through files
on disk (the manifest), so serialization must be stable and self-describing.

Everything reasons in the CANONICAL box format: normalized xyxy, top-left
origin, each coordinate in [0, 1] (see formats.py). Conversions to/from external
formats are delegated to formats.py so the math lives in exactly one place.

Import discipline: core must stay GPU-free and light — only pydantic + std lib
here. models.py MAY import formats.py; formats.py must NOT import models.py.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from grove.core import formats


# Small epsilon used when checking that clamping did not collapse a box. Boxes
# narrower/shorter than this after clamping are treated as degenerate.
_DEGENERATE_EPS = 1e-9


class BBox(BaseModel):
    """The canonical bounding box: normalized xyxy, top-left origin.

    Invariants (CLAUDE.md §6.1): 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1.
    On construction we CLAMP each coordinate into [0, 1] (detectors and the
    review UI can produce slightly-out-of-bounds values), then reject anything
    that is degenerate after clamping (x1 >= x2 or y1 >= y2).
    """

    x1: float
    y1: float
    x2: float
    y2: float

    @model_validator(mode="after")
    def _clamp_and_validate(self) -> "BBox":
        # Clamp first: a box a hair past the edge is a rounding artifact, not an
        # error — silently fixing it is friendlier than rejecting good data.
        self.x1 = _clamp01(self.x1)
        self.y1 = _clamp01(self.y1)
        self.x2 = _clamp01(self.x2)
        self.y2 = _clamp01(self.y2)
        # After clamping the box must still have positive area; a zero/negative
        # span means the box is meaningless and should never enter the dataset.
        if self.x2 - self.x1 <= _DEGENERATE_EPS:
            raise ValueError(
                f"degenerate BBox: x1 ({self.x1}) must be < x2 ({self.x2}) after clamping to [0,1]"
            )
        if self.y2 - self.y1 <= _DEGENERATE_EPS:
            raise ValueError(
                f"degenerate BBox: y1 ({self.y1}) must be < y2 ({self.y2}) after clamping to [0,1]"
            )
        return self

    # --- geometry (all in normalized units) ---------------------------------
    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    def clip(self) -> "BBox":
        """Return a clamped copy. (Construction already clamps, so this mainly
        exists as an explicit, intention-revealing API and a defensive copy.)"""
        return BBox(
            x1=_clamp01(self.x1),
            y1=_clamp01(self.y1),
            x2=_clamp01(self.x2),
            y2=_clamp01(self.y2),
        )

    # --- conversions out (delegate to formats.py) ---------------------------
    def to_yolo(self) -> tuple[float, float, float, float]:
        """-> (cx, cy, w, h), normalized."""
        return formats.canonical_to_yolo((self.x1, self.y1, self.x2, self.y2))

    def to_coco(self, W: int, H: int) -> tuple[float, float, float, float]:
        """-> (x_min, y_min, width, height) in absolute pixels."""
        return formats.canonical_to_coco((self.x1, self.y1, self.x2, self.y2), W, H)

    def to_pixel_xyxy(self, W: int, H: int) -> tuple[float, float, float, float]:
        """-> (x1, y1, x2, y2) in absolute pixels (for drawing)."""
        return formats.canonical_to_pixel_xyxy((self.x1, self.y1, self.x2, self.y2), W, H)

    # --- conversions in (classmethods, delegate to formats.py) --------------
    @classmethod
    def from_yolo(cls, cx: float, cy: float, w: float, h: float) -> "BBox":
        x1, y1, x2, y2 = formats.yolo_to_canonical((cx, cy, w, h))
        return cls(x1=x1, y1=y1, x2=x2, y2=y2)

    @classmethod
    def from_coco(cls, x_min: float, y_min: float, w: float, h: float, W: int, H: int) -> "BBox":
        x1, y1, x2, y2 = formats.coco_to_canonical((x_min, y_min, w, h), W, H)
        return cls(x1=x1, y1=y1, x2=x2, y2=y2)

    @classmethod
    def from_pixel_xyxy(cls, x1: float, y1: float, x2: float, y2: float, W: int, H: int) -> "BBox":
        nx1, ny1, nx2, ny2 = formats.pixel_xyxy_to_canonical((x1, y1, x2, y2), W, H)
        return cls(x1=nx1, y1=ny1, x2=nx2, y2=ny2)


class Detection(BaseModel):
    """One predicted/corrected box with its class label and optional score.

    score is None for human-drawn boxes (no confidence) and a float in [0,1]
    for detector output.
    """

    label: str
    box: BBox
    score: float | None = None


class ImageRecord(BaseModel):
    """One image and its current set of detections + review status.

    id is a stable identifier derived from the relative path at ingest time, so
    it survives the round trip through the dataset directory between machines.
    width/height are PIXEL dimensions (needed for canonical<->pixel conversion).
    """

    id: str
    path: str
    width: int
    height: int
    detections: list[Detection] = Field(default_factory=list)
    status: Literal["pending", "reviewed", "skipped"] = "pending"


class Manifest(BaseModel):
    """The on-disk record of one labeling run + its full provenance.

    CLAUDE.md §6.1/§12: "Every run must record its provenance" — so any dataset
    is traceable back to the exact settings that produced it, and two runs are
    never silently confused. These fields capture detector, prompt/ontology,
    thresholds, tiling, model + grove versions, and a timestamp.
    """

    images: list[ImageRecord] = Field(default_factory=list)

    # --- run provenance ------------------------------------------------------
    detector_backend: str
    ontology: dict[str, str]
    box_threshold: float
    text_threshold: float
    # Tiling settings: a dict of the resolved tiling config, or a bare bool when
    # only enabled/disabled is meaningful.
    tiling: dict | bool
    model_version: str | None = None
    grove_version: str
    # ISO-8601 UTC timestamp; defaults to "now" at construction time.
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # --- factory -------------------------------------------------------------
    @classmethod
    def new(
        cls,
        images: list[ImageRecord],
        *,
        detector_backend: str,
        ontology: dict[str, str],
        box_threshold: float,
        text_threshold: float,
        tiling: dict | bool,
        grove_version: str,
        model_version: str | None = None,
        created_at: str | None = None,
    ) -> "Manifest":
        """Build a Manifest, defaulting created_at to the current UTC time."""
        return cls(
            images=images,
            detector_backend=detector_backend,
            ontology=ontology,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            tiling=tiling,
            grove_version=grove_version,
            model_version=model_version,
            created_at=created_at or datetime.now(timezone.utc).isoformat(),
        )

    # --- persistence ---------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Write the manifest as pretty JSON (files on disk are the only
        interface between the GPU and local halves — keep it human-readable)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Manifest":
        """Read and validate a manifest from JSON on disk."""
        raw = Path(path).read_text(encoding="utf-8")
        return cls.model_validate(json.loads(raw))

    # --- lookup --------------------------------------------------------------
    def image_by_id(self) -> dict[str, ImageRecord]:
        """Build an id -> ImageRecord index for O(1) lookups."""
        return {img.id: img for img in self.images}

    def get(self, image_id: str) -> ImageRecord | None:
        """Return the ImageRecord with this id, or None if absent."""
        for img in self.images:
            if img.id == image_id:
                return img
        return None


def _clamp01(v: float) -> float:
    """Clamp a float into [0, 1]."""
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v
