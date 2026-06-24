"""Shared pytest fixtures + a GPU-free StubDetector (SPEC.md §13).

The whole point of this file is to make the pipeline and tiling wrapper testable
on a Mac with NO GPU and NONE of the heavy detector libs installed. Per §13 the
"detector backend can be stubbed in tests (return fixed canonical boxes) so the
pipeline and exporters are testable on the Mac without a GPU".

Import discipline for the test suite: NO cv2 / torch / supervision / autodistill /
ultralytics anywhere here. We synthesize "images" as plain numpy uint8 arrays
(exactly the HxWx3 BGR shape a Detector.detect receives) and build real on-disk
image files with Pillow (which IS installed). The StubDetector ignores pixel
content entirely and returns a fixed list of CANONICAL Detections, so its output
is fully deterministic and hand-checkable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from grove.core.config import Config
from grove.core.models import BBox, Detection


# ---------------------------------------------------------------------------
# StubDetector — implements the Detector protocol (base.Detector) without a GPU.
# ---------------------------------------------------------------------------
class StubDetector:
    """A Detector that returns a FIXED list of canonical Detections.

    Conforms to grove.pipeline.detectors.base.Detector: a single
    ``detect(image_bgr) -> list[Detection]`` method whose boxes are already in
    CANONICAL format (normalized xyxy, top-left origin, each coord in [0, 1]).

    The returned boxes are independent of the pixel content — that is the point:
    a deterministic teacher stand-in so tiling/translation/merge math can be
    asserted exactly (§13). ``detect`` returns FRESH BBox/Detection objects each
    call (deep copies) so a consumer that mutates results can't corrupt the
    template shared across tiles.
    """

    def __init__(self, detections: list[Detection] | None = None):
        # Default: one centered box, so a plain (un-tiled) call has something to
        # return. Tests that care about exact geometry pass their own list.
        if detections is None:
            detections = [
                Detection(label="mandarin", box=BBox(x1=0.25, y1=0.25, x2=0.75, y2=0.75), score=0.9)
            ]
        self._template = detections
        self.calls = 0  # how many times detect() was invoked (tile count probe)

    def detect(self, image_bgr) -> list[Detection]:  # noqa: ANN001 - numpy array
        self.calls += 1
        # Return independent copies so callers mutating boxes can't leak across
        # tile invocations (model_copy(deep=True) clones the nested BBox too).
        return [d.model_copy(deep=True) for d in self._template]


@pytest.fixture
def stub_detector() -> StubDetector:
    """A StubDetector with the default single centered box."""
    return StubDetector()


# ---------------------------------------------------------------------------
# Image helpers (numpy arrays + on-disk files via Pillow).
# ---------------------------------------------------------------------------
@pytest.fixture
def make_array_image():
    """Factory: build an HxWx3 uint8 BGR-shaped numpy array of given size.

    Note the numpy convention is (rows=H, cols=W, channels) — the same shape
    TiledDetector reads via image_bgr.shape[0]/[1]. We deliberately use a
    NON-square default so any W/H swap in code under test shows up.
    """

    def _make(width: int = 800, height: int = 600, value: int = 127) -> np.ndarray:
        return np.full((height, width, 3), value, dtype=np.uint8)

    return _make


@pytest.fixture
def make_image_file(tmp_path):
    """Factory: write a real image file to disk and return its Path.

    Builds a plain RGB image with Pillow (installed) so ingest/export can read it.
    Size is NON-square by default to catch coordinate transposition bugs (§12).
    """

    def _make(
        relpath: str = "img.jpg",
        width: int = 800,
        height: int = 600,
        color=(200, 120, 40),
        base_dir: Path | None = None,
    ) -> Path:
        root = base_dir if base_dir is not None else tmp_path
        dest = Path(root) / relpath
        dest.parent.mkdir(parents=True, exist_ok=True)
        img = Image.new("RGB", (width, height), color)
        img.save(dest)
        return dest

    return _make


# ---------------------------------------------------------------------------
# A real (light) Config rooted under a temp dir, per the task brief.
# ---------------------------------------------------------------------------
@pytest.fixture
def temp_config(tmp_path) -> Config:
    """A Config with input/work/export dirs under a fresh tmp_path.

    Built in code (not from YAML) so tests stay self-contained. work_dir and
    export_dir are created eagerly (mirrors load_config's ensure_dirs); input_dir
    is created here too so tests can drop fixture images straight in.
    """
    input_dir = tmp_path / "raw"
    work_dir = tmp_path / "work"
    export_dir = tmp_path / "dataset"
    input_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config.model_validate(
        {
            "project_name": "test-project",
            "paths": {
                "input_dir": str(input_dir),
                "work_dir": str(work_dir),
                "export_dir": str(export_dir),
            },
            "classes": ["mandarin"],
        }
    )
    cfg.paths.ensure_dirs()
    return cfg
