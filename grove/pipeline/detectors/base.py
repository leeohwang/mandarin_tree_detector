"""Detector abstraction + backend registry (SPEC.md §2.5, §6.4).

The whole point of this layer is the "swappable backend" requirement (§2.5):
the rest of the pipeline asks for boxes and never knows which open-vocab model
produced them. Each backend adapts its library's native, backend-specific
coordinate quirks (Grounding DINO / supervision emit absolute-pixel xyxy;
YOLO-World likewise) and returns ONLY canonical Detections — normalized xyxy,
top-left origin, each coord in [0, 1]. Nothing downstream sees pixels or
library-specific types (§6.4: "Do not let backend-specific coordinate quirks
leak past this layer").

IMPORTANT teacher-vs-student distinction (§2): these detectors are the slow,
accurate "teacher" used to *label* data — never the robot's deployable runtime.
Only the distilled YOLO student (pipeline/train.py) is robot-deployable.

Import discipline: this module must stay light so a Mac with only the [review]
extra can import it (it sits under pipeline/, but base.py itself pulls in NO
GPU/torch/autodistill code). The concrete backends DO import those heavy libs,
so get_detector() imports the chosen backend module LAZILY, inside the function.
Only typing + numpy(-for-typing) + core may be imported at module top.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from grove.core.models import Detection

if TYPE_CHECKING:
    # Imported only for type hints; never required at runtime, so importing
    # base.py stays cheap and dependency-free even if numpy is absent.
    import numpy as np

    from grove.core.config import Config


@runtime_checkable
class Detector(Protocol):
    """A pluggable open-vocab detector.

    The contract is identical for every backend: take one image, return its
    boxes already in CANONICAL format (normalized xyxy, top-left origin, each
    coordinate in [0, 1]). The caller (pipeline/detect.py, or the tiling wrapper)
    relies on this and stays backend-agnostic.
    """

    def detect(self, image_bgr: "np.ndarray") -> list[Detection]:
        """Detect objects in one image.

        Args:
            image_bgr: an HxWx3 uint8 numpy array in OpenCV BGR channel order
                (i.e. exactly what cv2.imread returns).

        Returns:
            A list of Detection, with each box in CANONICAL format (normalized
            xyxy, top-left origin, in [0, 1]). May be empty if nothing is found.
        """
        ...


# Registry of backend name -> import location. Values are (module, class_name)
# so we can import the heavy backend module LAZILY in get_detector() and keep
# this file free of torch/autodistill at import time. Keep keys in sync with
# DetectorConfig.backend (config.py) — the strict enum there is the first guard.
_BACKENDS: dict[str, tuple[str, str]] = {
    "grounding_dino": ("grove.pipeline.detectors.grounding_dino", "GroundingDINODetector"),
    # Same Grounding DINO model via HuggingFace transformers (pure-Python, runs on
    # a CPU-only Mac with no native build). The default "grounding_dino" backend
    # auto-falls back to this when autodistill is unavailable; this entry also lets
    # it be selected explicitly for exact provenance (§2.5).
    "grounding_dino_hf": ("grove.pipeline.detectors.grounding_dino_hf", "HFGroundingDINODetector"),
    "yolo_world": ("grove.pipeline.detectors.yolo_world", "YOLOWorldDetector"),
}


def get_detector(cfg: "Config") -> Detector:
    """Construct the detector chosen by config (the registry indirection).

    The backend module is imported here, lazily, so merely importing base.py
    never drags in torch/autodistill — essential for the light [review] install
    on the Mac (§3, §10). The chosen backend is built from the detector config:
    its ontology (prompt text -> class name), box_threshold and text_threshold.
    """
    backend = cfg.detector.backend
    try:
        module_path, class_name = _BACKENDS[backend]
    except KeyError:
        known = ", ".join(sorted(_BACKENDS))
        raise ValueError(
            f"unknown detector backend {backend!r}; known backends: {known}"
        ) from None

    # Lazy import: only now do we pull in the heavy library behind this backend.
    import importlib

    module = importlib.import_module(module_path)
    detector_cls = getattr(module, class_name)
    return detector_cls(
        ontology=cfg.detector.ontology,
        box_threshold=cfg.detector.box_threshold,
        text_threshold=cfg.detector.text_threshold,
    )
