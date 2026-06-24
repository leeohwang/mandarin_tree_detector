"""Typed, validated configuration for Grove (CLAUDE.md §6.2).

One YAML drives the whole pipeline. config.example.yaml is the shipped source of
truth (it targets individual trees); this module defines that same schema and
validates it with Pydantic v2, so a bad enum or out-of-range threshold fails
loudly at load time rather than producing a silently wrong dataset hours later.
The field defaults below are minimal fallbacks for keys omitted from the YAML,
not necessarily the shipped example's values.

Defaults are chosen so a MINIMAL YAML (just paths + ontology) still validates —
the operator edits only what matters (CLAUDE.md §10.4). load_config() also
auto-creates the output directories so the user never has to (§10.4).

Import discipline: std lib + pydantic + pyyaml only. No GPU/heavy deps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class PathsConfig(BaseModel):
    """Filesystem locations. Kept as strings so YAML stays simple; resolved to
    Path only where we actually touch the filesystem."""

    input_dir: str = "./data/raw"      # source images (scanned recursively)
    work_dir: str = "./data/work"      # manifest, predictions, previews
    export_dir: str = "./data/dataset" # final YOLO/COCO output

    def ensure_dirs(self) -> None:
        """Auto-create the OUTPUT dirs (work + export). We do NOT create
        input_dir — its absence is a real user error worth surfacing, not
        silently masking with an empty folder (CLAUDE.md §10.4)."""
        Path(self.work_dir).mkdir(parents=True, exist_ok=True)
        Path(self.export_dir).mkdir(parents=True, exist_ok=True)


class DetectorConfig(BaseModel):
    """Open-vocab detector backend + prompt ontology + thresholds (CLAUDE.md §6.4).

    ontology maps PROMPT TEXT -> CLASS NAME written into the labels. backend is a
    strict enum so a typo can't silently fall through to a default model.
    """

    backend: Literal["grounding_dino", "grounding_dino_hf", "yolo_world"] = "grounding_dino"
    ontology: dict[str, str] = Field(
        default_factory=lambda: {"mandarin orange on tree": "mandarin"}
    )
    box_threshold: float = 0.30
    text_threshold: float = 0.25
    # Safeguard against whole-image boxes: drop any detection whose box covers more
    # than this fraction of the image. Open-vocab detectors (especially on a whole-
    # scene prompt like "tree") frequently emit one box around the ENTIRE picture,
    # which is useless as an object-level label. 1.0 disables the safeguard (§11).
    max_box_area_frac: float = 0.85

    @field_validator("box_threshold", "text_threshold")
    @classmethod
    def _threshold_in_range(cls, v: float) -> float:
        # Confidence thresholds are probabilities; out-of-range values are bugs.
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {v}")
        return v

    @field_validator("max_box_area_frac")
    @classmethod
    def _area_frac_in_range(cls, v: float) -> float:
        # A fraction of image area; 0 is meaningless, 1.0 means "disabled".
        if not 0.0 < v <= 1.0:
            raise ValueError(f"max_box_area_frac must be in (0, 1], got {v}")
        return v


class TilingConfig(BaseModel):
    """SAHI-style tiling for small/distant fruit in large tree shots (§6.5)."""

    enabled: bool = False
    tile_size: int = 640
    overlap: float = 0.20
    nms_iou: float = 0.50

    @field_validator("overlap", "nms_iou")
    @classmethod
    def _fraction_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"value must be in [0, 1], got {v}")
        return v

    @field_validator("tile_size")
    @classmethod
    def _tile_size_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"tile_size must be > 0, got {v}")
        return v


class ExportConfig(BaseModel):
    """Dataset export: which formats, the deterministic train/val split (§6.7)."""

    formats: list[str] = Field(default_factory=lambda: ["yolo", "coco"])
    val_split: float = 0.15
    seed: int = 42  # seed the split so runs are reproducible (CLAUDE.md §12)

    @field_validator("val_split")
    @classmethod
    def _val_split_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"val_split must be in [0, 1], got {v}")
        return v

    @field_validator("formats")
    @classmethod
    def _known_formats(cls, v: list[str]) -> list[str]:
        allowed = {"yolo", "coco"}
        bad = [f for f in v if f not in allowed]
        if bad:
            raise ValueError(f"unknown export format(s) {bad}; allowed: {sorted(allowed)}")
        if not v:
            raise ValueError("export.formats must list at least one format")
        return v


class TrainConfig(BaseModel):
    """Optional student-distillation stage (GPU side, §6.9)."""

    enabled: bool = False
    model: str = "yolov8n.pt"
    epochs: int = 100
    imgsz: int = 640

    @field_validator("epochs", "imgsz")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"value must be > 0, got {v}")
        return v


class ReviewConfig(BaseModel):
    """Local FastAPI review server bind address (§6.8)."""

    host: str = "127.0.0.1"
    port: int = 8000

    @field_validator("port")
    @classmethod
    def _valid_port(cls, v: int) -> int:
        if not 1 <= v <= 65535:
            raise ValueError(f"port must be in [1, 65535], got {v}")
        return v


class Config(BaseModel):
    """Top-level config. Every nested section has defaults, so a minimal YAML
    validates and the operator edits only paths + ontology."""

    project_name: str = "mandarin-orchard-v1"
    paths: PathsConfig = Field(default_factory=PathsConfig)
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    tiling: TilingConfig = Field(default_factory=TilingConfig)
    classes: list[str] = Field(default_factory=lambda: ["mandarin"])
    export: ExportConfig = Field(default_factory=ExportConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    review: ReviewConfig = Field(default_factory=ReviewConfig)

    @field_validator("classes")
    @classmethod
    def _classes_non_empty(cls, v: list[str]) -> list[str]:
        # YOLO needs a stable, non-empty class list (index = class id, §6.2).
        if not v:
            raise ValueError("classes must list at least one class name")
        return v


def load_config(path: str | Path) -> Config:
    """Read YAML, validate into Config, auto-create output dirs, and return it.

    Validation is strict: bad enums / out-of-range thresholds raise clear errors
    here rather than corrupting a dataset downstream.
    """
    raw = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(raw) or {}  # empty file -> {} -> all defaults
    if not isinstance(data, dict):
        raise ValueError(f"config root must be a mapping, got {type(data).__name__}")
    cfg = Config.model_validate(data)
    cfg.paths.ensure_dirs()  # auto-create outputs (CLAUDE.md §10.4)
    return cfg
