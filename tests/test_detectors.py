"""Executed validation of the GPU detector adapters' supervision -> canonical
conversion (SPEC.md §2.4, §6.4, §12).

A real Grounding DINO / YOLO-World forward pass needs a GPU and is out of scope
for CI, but the *coordinate + label conversion* — the silent-bug surface — does
NOT need a model. Here we bypass each backend's heavy __init__ (object.__new__),
inject a fake model whose .predict returns a REAL `supervision.Detections`, and
assert the adapter maps absolute-pixel xyxy -> canonical normalized boxes and
class_id -> class name correctly.

Skipped automatically where `supervision` is not installed (e.g. a review-only
Mac with just the [review] extra) — these only need to run where the GPU stack,
hence supervision, is present.
"""
from __future__ import annotations

import numpy as np
import pytest

sv = pytest.importorskip("supervision")

from grove.core.models import Detection  # noqa: E402
from grove.pipeline.detectors.grounding_dino import GroundingDINODetector  # noqa: E402
from grove.pipeline.detectors.yolo_world import YOLOWorldDetector  # noqa: E402

BACKENDS = [GroundingDINODetector, YOLOWorldDetector]


class _FakeModel:
    """Stand-in for an autodistill BaseModel: .predict returns fixed Detections."""

    def __init__(self, detections):
        self._detections = detections

    def predict(self, image, *args, **kwargs):  # accept confidence= kwarg etc.
        return self._detections


def _make(cls, classes, sv_detections):
    """Build a backend instance WITHOUT importing autodistill/torch."""
    obj = object.__new__(cls)
    obj._classes = list(classes)
    obj._model = _FakeModel(sv_detections)
    obj.box_threshold = 0.0  # read by YOLOWorldDetector.detect
    return obj


def _img(w, h):
    """An HxWx3 BGR-style uint8 array (only its shape matters here)."""
    return np.zeros((h, w, 3), dtype=np.uint8)


@pytest.mark.parametrize("cls", BACKENDS)
def test_pixel_xyxy_to_canonical_nonsquare(cls):
    # Non-square image so any W/H swap surfaces. Box = right half, bottom half:
    # pixels x in [400,800] of W=800 -> [0.5,1.0]; y in [200,400] of H=400 -> [0.5,1.0].
    W, H = 800, 400
    det = _make(
        cls, ["mandarin"],
        sv.Detections(
            xyxy=np.array([[400.0, 200.0, 800.0, 400.0]]),
            confidence=np.array([0.83]),
            class_id=np.array([0]),
        ),
    )
    out = det.detect(_img(W, H))
    assert len(out) == 1
    d = out[0]
    assert isinstance(d, Detection)
    assert d.label == "mandarin"
    assert d.score == pytest.approx(0.83)
    b = d.box
    assert (b.x1, b.x2) == pytest.approx((0.5, 1.0))
    assert (b.y1, b.y2) == pytest.approx((0.5, 1.0))
    # A W/H transposition would make x1 == 400/400 == 1.0 (degenerate) — caught.


@pytest.mark.parametrize("cls", BACKENDS)
def test_class_id_maps_to_name_in_order(cls):
    # class_id indexes the ordered class list: [1,0] -> ["b","a"].
    det = _make(
        cls, ["a", "b"],
        sv.Detections(
            xyxy=np.array([[0.0, 0.0, 50.0, 50.0], [50.0, 50.0, 100.0, 100.0]]),
            confidence=np.array([0.9, 0.7]),
            class_id=np.array([1, 0]),
        ),
    )
    out = det.detect(_img(100, 100))
    assert [d.label for d in out] == ["b", "a"]


@pytest.mark.parametrize("cls", BACKENDS)
def test_out_of_range_class_id_is_dropped(cls):
    # Only one class but model reports class_id=5 -> drop, don't mislabel (§12).
    det = _make(
        cls, ["mandarin"],
        sv.Detections(
            xyxy=np.array([[10.0, 10.0, 20.0, 20.0]]),
            confidence=np.array([0.5]),
            class_id=np.array([5]),
        ),
    )
    assert det.detect(_img(100, 100)) == []


@pytest.mark.parametrize("cls", BACKENDS)
def test_degenerate_box_is_dropped(cls):
    # Zero-width box (x1 == x2) must be dropped, not crash the batch.
    det = _make(
        cls, ["mandarin"],
        sv.Detections(
            xyxy=np.array([[10.0, 10.0, 10.0, 40.0]]),
            confidence=np.array([0.5]),
            class_id=np.array([0]),
        ),
    )
    assert det.detect(_img(100, 100)) == []


@pytest.mark.parametrize("cls", BACKENDS)
def test_empty_detections(cls):
    det = _make(
        cls, ["mandarin"],
        sv.Detections(xyxy=np.empty((0, 4)), confidence=np.empty((0,)), class_id=np.empty((0,), dtype=int)),
    )
    assert det.detect(_img(640, 480)) == []
