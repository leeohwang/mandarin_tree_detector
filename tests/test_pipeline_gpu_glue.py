"""Executed validation of the GPU-side pipeline GLUE that needs cv2/supervision
but NOT a real model or a GPU (SPEC.md §6.4, §6.6, §7).

We stub the detector (the only part that truly needs a GPU) and run the real
detect() orchestration + resumability, and the real annotate() drawing. These
were previously only syntax-checked; here they actually execute.

Auto-skipped where cv2 / supervision are absent (e.g. a review-only Mac).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from PIL import Image  # noqa: E402

from grove.core.config import Config  # noqa: E402
from grove.core.models import BBox, Detection, Manifest  # noqa: E402


def _cfg(tmp_path, tiling=False):
    raw = tmp_path / "raw"
    raw.mkdir()
    # Two NON-SQUARE images so any axis confusion in the glue would surface.
    Image.new("RGB", (200, 120), (120, 60, 30)).save(raw / "a.jpg")
    Image.new("RGB", (160, 240), (30, 120, 60)).save(raw / "b.jpg")
    return Config.model_validate({
        "paths": {
            "input_dir": str(raw),
            "work_dir": str(tmp_path / "work"),
            "export_dir": str(tmp_path / "ds"),
        },
        "classes": ["mandarin"],
        "tiling": {"enabled": tiling},
    })


class _StubDetector:
    """Returns one fixed canonical detection per image; counts calls."""

    def __init__(self):
        self.calls = 0

    def detect(self, image_bgr):
        self.calls += 1
        return [Detection(label="mandarin", box=BBox(x1=0.1, y1=0.1, x2=0.4, y2=0.5), score=0.8)]


def test_detect_drops_oversized_boxes(tmp_path, monkeypatch):
    """Whole-image-box safeguard: detect() drops any box whose area exceeds
    cfg.detector.max_box_area_frac — the common open-vocab failure mode where the
    detector boxes the ENTIRE image (SPEC.md §11)."""
    from grove.pipeline import detect as detect_mod
    from grove.pipeline.ingest import ingest

    cfg = _cfg(tmp_path)
    cfg.detector.max_box_area_frac = 0.85
    ingest(cfg)

    class _OversizeStub:
        # One ~full-frame box (area ~1.0 -> dropped) + one legit box (kept).
        def detect(self, image_bgr):
            return [
                Detection(label="tree", box=BBox(x1=0.001, y1=0.001, x2=0.999, y2=0.999), score=0.9),
                Detection(label="tree", box=BBox(x1=0.10, y1=0.10, x2=0.40, y2=0.50), score=0.8),
            ]

    monkeypatch.setattr(detect_mod, "get_detector", lambda c: _OversizeStub())
    man = detect_mod.detect(cfg)
    for r in man.images:
        assert len(r.detections) == 1, "full-frame box should have been dropped"
        assert r.detections[0].box.area <= 0.85


def test_detect_orchestration_and_provenance(tmp_path, monkeypatch):
    from grove.pipeline import detect as detect_mod
    from grove.pipeline.ingest import ingest

    cfg = _cfg(tmp_path)
    cfg.detector.box_threshold = 0.42  # to prove provenance comes from live cfg
    ingest(cfg)

    stub = _StubDetector()
    monkeypatch.setattr(detect_mod, "get_detector", lambda c: stub)
    man = detect_mod.detect(cfg)

    assert stub.calls == 2
    assert all(len(r.detections) == 1 for r in man.images)
    # The detect() provenance fix: settings recorded from the LIVE cfg, not the
    # ingest manifest.
    pred = Manifest.load(Path(cfg.paths.work_dir) / "predictions.json")
    assert pred.box_threshold == pytest.approx(0.42)
    assert pred.detector_backend == cfg.detector.backend


def test_detect_is_resumable(tmp_path, monkeypatch):
    from grove.pipeline import detect as detect_mod
    from grove.pipeline.ingest import ingest

    cfg = _cfg(tmp_path)
    ingest(cfg)

    stub = _StubDetector()
    monkeypatch.setattr(detect_mod, "get_detector", lambda c: stub)
    detect_mod.detect(cfg)
    assert stub.calls == 2

    # Second run: every image was already processed, so the detector must NOT be
    # called again (the whole point of resumability — don't burn GPU quota, §7/§10).
    class _Boom:
        def detect(self, image_bgr):
            raise AssertionError("resumed image was re-detected")

    monkeypatch.setattr(detect_mod, "get_detector", lambda c: _Boom())
    man2 = detect_mod.detect(cfg)
    assert all(len(r.detections) == 1 for r in man2.images)  # carried forward


def test_annotate_actually_draws_boxes(tmp_path, monkeypatch):
    """The critical annotate fix: with ColorLookup.INDEX, boxes RENDER. If the
    old ColorLookup.CLASS bug were present, annotate would silently draw nothing
    and the preview would be byte-identical to the prepared image."""
    pytest.importorskip("supervision")
    from grove.pipeline import detect as detect_mod
    from grove.pipeline.annotate import annotate
    from grove.pipeline.ingest import ingest

    cfg = _cfg(tmp_path)
    ingest(cfg)
    monkeypatch.setattr(detect_mod, "get_detector", lambda c: _StubDetector())
    detect_mod.detect(cfg)

    annotate(cfg)

    previews = Path(cfg.paths.work_dir) / "previews"
    files = sorted(previews.glob("*"))
    assert len(files) == 2, files

    man = Manifest.load(Path(cfg.paths.work_dir) / "predictions.json")
    for rec in man.images:
        name = Path(rec.path).name  # previews/<id>.<ext>
        preview = cv2.imread(str(previews / name))
        source = cv2.imread(str(Path(cfg.paths.work_dir) / rec.path))
        assert preview is not None and source is not None
        assert preview.shape == source.shape
        # A drawn box changes pixels -> the preview must differ from the source.
        assert not np.array_equal(preview, source), f"annotate drew nothing on {rec.id}"


def test_train_passes_absolute_project(tmp_path, monkeypatch):
    """Regression: ultralytics re-roots a RELATIVE `project` under its own runs_dir,
    scattering student weights outside work_dir. train() must pass an ABSOLUTE
    project == <work_dir>/runs (found by actually running train on CPU)."""
    import sys
    import types

    from grove.core.config import Config
    from grove.pipeline.train import train

    captured = {}

    class _FakeYOLO:
        def __init__(self, model):
            self.model = model

        def train(self, **kw):
            captured.update(kw)
            save_dir = Path(kw["project"]) / kw["name"]
            (save_dir / "weights").mkdir(parents=True, exist_ok=True)
            return types.SimpleNamespace(save_dir=save_dir)

    fake = types.ModuleType("ultralytics")
    fake.YOLO = _FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", fake)

    work = tmp_path / "work"
    export = tmp_path / "ds"
    export.mkdir(parents=True)
    (export / "data.yaml").write_text("nc: 1\nnames: [mandarin]\n")
    cfg = Config.model_validate({
        "paths": {"input_dir": str(tmp_path / "raw"), "work_dir": str(work), "export_dir": str(export)},
        "classes": ["mandarin"],
        "train": {"enabled": True, "model": "yolov8n.pt", "epochs": 1, "imgsz": 320},
    })

    train(cfg)

    project = Path(captured["project"])
    assert project.is_absolute(), captured["project"]
    assert project == (work / "runs").resolve()
