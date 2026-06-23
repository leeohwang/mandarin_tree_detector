"""Unit tests for the HuggingFace Grounding DINO backend's pure helpers.

A real forward pass needs the model weights and is out of scope for CI, but the
label-mapping and model-source resolution are pure logic. We build instances via
``object.__new__`` (bypassing the heavy __init__/model load) and exercise only
those helpers — mirroring tests/test_detectors.py's approach.

The module's heavy imports (torch/transformers) are lazy (inside __init__), so
importing it here is light and needs no GPU stack.
"""
from __future__ import annotations

from grove.pipeline.detectors.grounding_dino_hf import (
    _DEFAULT_MODEL,
    HFGroundingDINODetector,
)


def _make(classes, phrase_to_class, nms_iou=0.5):
    """Build a detector WITHOUT loading torch/transformers or any weights."""
    obj = object.__new__(HFGroundingDINODetector)
    obj._classes = list(classes)
    obj._phrase_to_class = dict(phrase_to_class)
    obj._nms_iou = nms_iou
    return obj


def test_map_label_exact_and_period_stripped():
    d = _make(["tree"], {"tree": "tree"})
    assert d._map_label("tree") == "tree"
    assert d._map_label("tree.") == "tree"
    assert d._map_label("  Tree . ") == "tree"  # normalized: strip/lower/rstrip


def test_map_label_word_overlap_and_fallback():
    d = _make(["tree"], {"tree": "tree"})
    assert d._map_label("a tall tree") == "tree"  # token overlap
    assert d._map_label("") == "tree"  # empty -> first class
    assert d._map_label("railroad") == "tree"  # unknown -> first class fallback


def test_map_label_multiclass_substring_wins_right_class():
    d = _make(
        ["mandarin", "leaf"],
        {"mandarin orange": "mandarin", "leaf": "leaf"},
    )
    assert d._map_label("mandarin orange") == "mandarin"
    assert d._map_label("orange") == "mandarin"  # substring of "mandarin orange"
    assert d._map_label("leaf") == "leaf"


def test_resolve_model_source_env_override(monkeypatch):
    # Explicit override always wins (an HF id or a local dir path).
    monkeypatch.setenv("GROVE_GDINO_HF_MODEL", "/tmp/some/local/model")
    assert HFGroundingDINODetector._resolve_model_source() == "/tmp/some/local/model"


def test_resolve_model_source_default_when_no_env_and_no_cache(monkeypatch, tmp_path):
    # No override and no local cache -> the default HF hub id. Point HOME at an
    # empty tmp dir so the real ~/.cache copy on the dev box can't leak in.
    monkeypatch.delenv("GROVE_GDINO_HF_MODEL", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert HFGroundingDINODetector._resolve_model_source() == _DEFAULT_MODEL
