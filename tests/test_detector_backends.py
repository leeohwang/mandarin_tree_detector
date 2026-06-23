"""Construction, registry dispatch, and fallback for the GPU detector backends
(CLAUDE.md §2.5, §6.4).

`tests/test_detectors.py` validates the *coordinate conversion* inside each
backend's ``detect()`` but bypasses construction (``object.__new__`` + a hand-set
``_classes``/``_model``). That leaves the parts that only run on the GPU side —
each backend's real ``__init__`` (building a ``CaptionOntology``, instantiating the
underlying model, extracting the ordered class list) and the ``get_detector()``
registry dispatch — completely unexercised, because ``autodistill`` cannot be
installed on the Intel Mac dev box (no CUDA build).

We close that gap WITHOUT a GPU by injecting fake ``autodistill`` /
``autodistill_grounding_dino`` / ``autodistill_yolo_world`` modules into
``sys.modules``. The backends import those lazily *inside* ``__init__``, so the
fakes are picked up at construction time. The fake models' ``.predict`` returns a
REAL ``supervision.Detections`` (absolute-pixel xyxy), so the full path —
config -> get_detector -> real __init__ -> detect -> canonical boxes — runs end to
end. This is the closest thing to the Kaggle/autodistill path that is reachable on
a CPU-only machine, and it pins:
  * the registry maps each backend string to the right class (base.py),
  * each real ``__init__`` wires ontology + thresholds through correctly,
  * ``CaptionOntology(prompt->class)`` ordering becomes the class-id mapping,
  * the ``grounding_dino`` backend transparently falls back to the HF backend when
    autodistill is absent (the actual situation on the Mac).

Skipped where ``supervision`` is absent (review-only [review] install).
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest

sv = pytest.importorskip("supervision")

from grove.core.config import Config  # noqa: E402
from grove.core.models import BBox, Detection  # noqa: E402
from grove.pipeline.detectors import base  # noqa: E402
from grove.pipeline.detectors.base import get_detector  # noqa: E402


# ---------------------------------------------------------------------------
# Fakes standing in for the autodistill library surface the backends touch.
# ---------------------------------------------------------------------------
class _FakeCaptionOntology:
    """Mimics ``autodistill.detection.CaptionOntology``.

    Real CaptionOntology is built from a {prompt_text: class_name} dict and
    exposes the output class names, in id order, via ``.classes()`` — which is
    exactly what both backends rely on to map class_id -> name.
    """

    def __init__(self, mapping):
        self.mapping = dict(mapping)

    def classes(self):
        return list(self.mapping.values())

    def prompts(self):
        return list(self.mapping.keys())


def _one_box_detections():
    """A real supervision.Detections: one box, right+bottom quadrant of an
    800x400 image (x in [400,800] -> [0.5,1.0]; y in [200,400] -> [0.5,1.0])."""
    return sv.Detections(
        xyxy=np.array([[400.0, 200.0, 800.0, 400.0]]),
        confidence=np.array([0.83]),
        class_id=np.array([0]),
    )


class _RecordingGroundingDINO:
    """Stands in for ``autodistill_grounding_dino.GroundingDINO``; records the
    ctor kwargs so we can assert config flowed through, and what .predict saw."""

    instances: list["_RecordingGroundingDINO"] = []

    def __init__(self, ontology, box_threshold, text_threshold):
        self.ontology = ontology
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.predict_calls: list = []
        _RecordingGroundingDINO.instances.append(self)

    def predict(self, image):  # GroundingDINO backend calls predict(image)
        self.predict_calls.append(image)
        return _one_box_detections()


class _RecordingYOLOWorld:
    """Stands in for ``autodistill_yolo_world.YOLOWorld`` (ctor takes only
    ``ontology``; predict takes a ``confidence`` kwarg)."""

    instances: list["_RecordingYOLOWorld"] = []

    def __init__(self, ontology):
        self.ontology = ontology
        self.predict_calls: list = []
        _RecordingYOLOWorld.instances.append(self)

    def predict(self, image, confidence=None):
        self.predict_calls.append((image, confidence))
        return _one_box_detections()


@pytest.fixture
def fake_autodistill(monkeypatch):
    """Install fake autodistill modules so the backends' lazy imports resolve.

    Cleaned up automatically by monkeypatch, so it never leaks into other tests
    (e.g. the fallback test below, which needs autodistill to be ABSENT).
    """
    autodistill = types.ModuleType("autodistill")
    detection = types.ModuleType("autodistill.detection")
    detection.CaptionOntology = _FakeCaptionOntology
    autodistill.detection = detection  # so `import autodistill.detection` resolves

    gd = types.ModuleType("autodistill_grounding_dino")
    gd.GroundingDINO = _RecordingGroundingDINO
    yw = types.ModuleType("autodistill_yolo_world")
    yw.YOLOWorld = _RecordingYOLOWorld

    monkeypatch.setitem(sys.modules, "autodistill", autodistill)
    monkeypatch.setitem(sys.modules, "autodistill.detection", detection)
    monkeypatch.setitem(sys.modules, "autodistill_grounding_dino", gd)
    monkeypatch.setitem(sys.modules, "autodistill_yolo_world", yw)

    _RecordingGroundingDINO.instances.clear()
    _RecordingYOLOWorld.instances.clear()
    yield


def _img(w, h):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _cfg(backend: str, ontology: dict[str, str], box=0.30, text=0.25) -> Config:
    return Config.model_validate(
        {
            "detector": {
                "backend": backend,
                "ontology": ontology,
                "box_threshold": box,
                "text_threshold": text,
            },
            "classes": list(dict.fromkeys(ontology.values())),
        }
    )


# ---------------------------------------------------------------------------
# Registry dispatch (base.py) — independent of any heavy lib.
# ---------------------------------------------------------------------------
def test_unknown_backend_raises():
    # The config enum is the first guard; bypass it (construct DetectorConfig
    # then poke the attr) to prove get_detector ALSO guards, with a helpful msg.
    cfg = _cfg("yolo_world", {"x": "mandarin"})
    object.__setattr__(cfg.detector, "backend", "does_not_exist")
    with pytest.raises(ValueError, match="unknown detector backend"):
        get_detector(cfg)


def test_registry_lists_all_documented_backends():
    # Keys must stay in sync with the DetectorConfig.backend enum (§6.4).
    assert set(base._BACKENDS) == {"grounding_dino", "grounding_dino_hf", "yolo_world"}


# ---------------------------------------------------------------------------
# Grounding DINO via the autodistill path (the Kaggle/GPU construction path).
# ---------------------------------------------------------------------------
def test_grounding_dino_autodistill_construction_and_detect(fake_autodistill):
    cfg = _cfg("grounding_dino", {"mandarin orange on tree": "mandarin"}, box=0.42, text=0.27)

    det = get_detector(cfg)

    # Real __init__ ran the autodistill branch: a model was built (no HF delegate).
    assert getattr(det, "_delegate", None) is None
    assert len(_RecordingGroundingDINO.instances) == 1
    model = _RecordingGroundingDINO.instances[0]
    # Thresholds from config flowed into the underlying model.
    assert model.box_threshold == pytest.approx(0.42)
    assert model.text_threshold == pytest.approx(0.27)
    # Ontology became a CaptionOntology with the configured prompt->class mapping.
    assert isinstance(model.ontology, _FakeCaptionOntology)
    assert model.ontology.mapping == {"mandarin orange on tree": "mandarin"}
    # The backend cached the ordered class list off the ontology.
    assert det._classes == ["mandarin"]

    # End-to-end detect: absolute-pixel supervision box -> canonical normalized.
    out = det.detect(_img(800, 400))
    assert len(out) == 1 and isinstance(out[0], Detection)
    assert out[0].label == "mandarin"
    assert out[0].score == pytest.approx(0.83)
    b = out[0].box
    assert (b.x1, b.y1, b.x2, b.y2) == pytest.approx((0.5, 0.5, 1.0, 1.0))


def test_grounding_dino_ontology_order_drives_class_ids(fake_autodistill):
    # Two-class ontology: insertion order defines class_id 0,1 (§6.2).
    cfg = _cfg("grounding_dino", {"ripe mandarin": "ripe", "green mandarin": "unripe"})
    det = get_detector(cfg)
    assert det._classes == ["ripe", "unripe"]

    # Feed class_id 1 then 0 and confirm name mapping follows ontology order.
    model = _RecordingGroundingDINO.instances[0]
    model.predict = lambda image: sv.Detections(
        xyxy=np.array([[0.0, 0.0, 80.0, 80.0], [80.0, 80.0, 160.0, 160.0]]),
        confidence=np.array([0.9, 0.6]),
        class_id=np.array([1, 0]),
    )
    out = det.detect(_img(160, 160))
    assert [d.label for d in out] == ["unripe", "ripe"]


# ---------------------------------------------------------------------------
# YOLO-World via autodistill — proves the swappable-backend abstraction (§2.5).
# ---------------------------------------------------------------------------
def test_yolo_world_construction_passes_confidence_and_converts(fake_autodistill):
    cfg = _cfg("yolo_world", {"orange fruit": "mandarin"}, box=0.35, text=0.20)

    det = get_detector(cfg)

    assert len(_RecordingYOLOWorld.instances) == 1
    model = _RecordingYOLOWorld.instances[0]
    assert isinstance(model.ontology, _FakeCaptionOntology)
    assert model.ontology.mapping == {"orange fruit": "mandarin"}
    assert det._classes == ["mandarin"]

    out = det.detect(_img(800, 400))

    # YOLO-World has no separate text threshold; box_threshold is forwarded as the
    # predict `confidence` filter (the one backend-specific quirk this layer hides).
    assert model.predict_calls and model.predict_calls[0][1] == pytest.approx(0.35)
    assert len(out) == 1
    b = out[0].box
    assert (b.x1, b.y1, b.x2, b.y2) == pytest.approx((0.5, 0.5, 1.0, 1.0))


def test_both_backends_indistinguishable_from_outside(fake_autodistill):
    # Same image + equivalent single-box model output -> identical canonical
    # Detection regardless of which backend produced it (§6.4 contract).
    gd = get_detector(_cfg("grounding_dino", {"orange fruit": "mandarin"}))
    yw = get_detector(_cfg("yolo_world", {"orange fruit": "mandarin"}))
    gout = gd.detect(_img(800, 400))
    yout = yw.detect(_img(800, 400))
    assert len(gout) == len(yout) == 1
    assert gout[0].label == yout[0].label == "mandarin"
    gb, yb = gout[0].box, yout[0].box
    assert (gb.x1, gb.y1, gb.x2, gb.y2) == pytest.approx((yb.x1, yb.y1, yb.x2, yb.y2))


# ---------------------------------------------------------------------------
# Fallback: `grounding_dino` -> HF backend when autodistill is absent.
# This is the path actually taken on the Intel Mac (no autodistill installed).
# ---------------------------------------------------------------------------
def test_grounding_dino_falls_back_to_hf_when_autodistill_absent(monkeypatch):
    # Guard: autodistill genuinely not importable here (no fake_autodistill).
    assert "autodistill" not in sys.modules or sys.modules["autodistill"] is None

    captured = {}

    class _FakeHF:
        def __init__(self, ontology, box_threshold, text_threshold):
            captured["args"] = (ontology, box_threshold, text_threshold)

        def detect(self, image_bgr):
            # HF backend already returns CANONICAL Detections; the autodistill
            # wrapper must forward them through untouched.
            return [
                Detection(
                    label="mandarin",
                    box=BBox(x1=0.1, y1=0.1, x2=0.2, y2=0.2),
                    score=0.5,
                )
            ]

    # The fallback does `from ...grounding_dino_hf import HFGroundingDINODetector`
    # at call time, so patching the module attribute is enough.
    import grove.pipeline.detectors.grounding_dino_hf as hf_mod

    monkeypatch.setattr(hf_mod, "HFGroundingDINODetector", _FakeHF)

    cfg = _cfg("grounding_dino", {"tree trunk": "mandarin"}, box=0.20, text=0.15)
    det = get_detector(cfg)

    # Delegate is the HF backend, built with the SAME ontology + thresholds.
    assert getattr(det, "_delegate", None) is not None
    assert isinstance(det._delegate, _FakeHF)
    assert captured["args"] == ({"tree trunk": "mandarin"}, 0.20, 0.15)

    # detect() transparently forwards to the delegate.
    out = det.detect(_img(100, 100))
    assert len(out) == 1 and out[0].label == "mandarin"
