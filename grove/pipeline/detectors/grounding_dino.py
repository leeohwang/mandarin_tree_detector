"""Grounding DINO detector backend (SPEC.md §2.1, §6.4).

================================  READ THIS  ================================
Grounding DINO is the TEACHER / auto-labeler in this pipeline. It is accurate
but SLOW — tens of seconds per image on CPU, and even on a GPU it is far too
heavy for a robot's real-time camera. It exists ONLY to draft labels that a
human then corrects (the review step is mandatory). It is NOT the robot's
runtime detector. The deployable, real-time model is the distilled YOLO
"student" trained from these reviewed labels (see pipeline/train.py). Never
treat this class as something that runs on the robot. (SPEC.md §2.1, §12)
============================================================================

This backend wraps `autodistill-grounding-dino`. From the outside it satisfies
the Detector protocol (pipeline/detectors/base.py): take an image, return a
list[Detection] whose boxes are already in the CANONICAL format (normalized
xyxy, top-left origin). All of Grounding DINO's coordinate quirks — namely that
`supervision` reports `xyxy` in ABSOLUTE PIXELS — are normalized away here so
nothing downstream ever sees backend-specific coordinates (SPEC.md §2.4, §6.4).

Heavy imports (autodistill / torch) are LAZY: they happen inside __init__, so a
light consumer (e.g. the CLI on a review-only Mac install) can import this module
without those packages present. Only instantiating GroundingDINODetector pulls
them in.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from grove.core.models import BBox, Detection

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    import numpy as np

logger = logging.getLogger(__name__)


class GroundingDINODetector:
    """Open-vocabulary detector backed by autodistill's Grounding DINO.

    TEACHER MODEL — auto-labeling only, never the robot runtime (see module
    docstring). Construct it with an ontology mapping prompt text -> class name
    plus the box/text confidence thresholds, then call .detect(image_bgr).
    """

    def __init__(
        self,
        ontology: dict[str, str],
        box_threshold: float,
        text_threshold: float,
    ) -> None:
        # --- LAZY heavy imports -------------------------------------------------
        # autodistill + autodistill_grounding_dino drag in torch and download
        # model weights on first use. Importing them here (not at module top)
        # keeps this file importable on a GPU-free Mac with only the [review]
        # extra installed, where it is merely syntax-checked, never instantiated.
        #
        # FALLBACK: autodistill-grounding-dino pulls in groundingdino-py, which
        # compiles native CUDA/C++ ops and is impractical on the Intel Mac dev box
        # (no CUDA). When it is not importable, transparently use the SAME Grounding
        # DINO model via HuggingFace transformers (detectors/grounding_dino_hf.py),
        # which is pure-Python and CPU-friendly. The Kaggle/GPU path (autodistill
        # present) is unchanged; locally, `grove detect` still runs Grounding DINO.
        try:
            from autodistill.detection import CaptionOntology
            from autodistill_grounding_dino import GroundingDINO
        except ImportError as exc:
            from grove.pipeline.detectors.grounding_dino_hf import HFGroundingDINODetector

            logger.info(
                "autodistill Grounding DINO unavailable (%s); falling back to the "
                "HuggingFace transformers Grounding DINO (same model, CPU-friendly, "
                "no native build).",
                exc,
            )
            self._delegate: HFGroundingDINODetector | None = HFGroundingDINODetector(
                ontology, box_threshold, text_threshold
            )
            return

        # autodistill path: no delegate, run the model built below directly.
        self._delegate = None

        # CaptionOntology maps PROMPT TEXT -> CLASS NAME. The class name is what
        # gets written into the exported labels; the prompt is what the model is
        # actually conditioned on. ontology dict order defines class_id ordering.
        self._ontology = CaptionOntology(ontology)

        # Construct the model. Weights AUTO-DOWNLOAD on first construction/use —
        # the operator is never asked to fetch them (SPEC.md §10.4).
        self._model = GroundingDINO(
            ontology=self._ontology,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        # Cache the class-name list once. CaptionOntology.classes() returns the
        # class names in the same order autodistill assigns class_id values, so
        # class_id is a direct index into this list.
        self._classes: list[str] = list(self._ontology.classes())

        logger.info(
            "GroundingDINODetector ready (TEACHER/labeler, not robot runtime): "
            "classes=%s box_threshold=%.3f text_threshold=%.3f",
            self._classes,
            box_threshold,
            text_threshold,
        )

    def detect(self, image_bgr: "np.ndarray") -> list[Detection]:
        """Run Grounding DINO on one image and return canonical Detections.

        `image_bgr` is an OpenCV-style HxWx3 ndarray in BGR order (the project's
        in-memory image convention). autodistill's GroundingDINO.predict accepts
        an ndarray directly and internally treats it as a cv2/BGR image (it is
        built on supervision/cv2 conventions), so we pass the array through
        as-is — no manual BGR->RGB swap here. We hand the raw BGR ndarray straight
        to the model rather than writing a temp file, which avoids extra disk I/O.

        The returned `supervision.Detections` has:
          - xyxy:        ABSOLUTE PIXEL corners (x1, y1, x2, y2)
          - class_id:    index into self._classes (the ontology class order)
          - confidence:  per-detection score in [0, 1]
        We convert each xyxy box to canonical via BBox.from_pixel_xyxy using the
        image's pixel W/H (taken from image_bgr.shape), and map class_id ->
        class name through self._classes.
        """
        # If autodistill was unavailable we delegate to the HuggingFace Grounding
        # DINO backend, which already returns canonical Detections. getattr keeps
        # this safe for test doubles that build the object via object.__new__ and
        # never set _delegate.
        if getattr(self, "_delegate", None) is not None:
            return self._delegate.detect(image_bgr)

        # Pixel dimensions from the ndarray: shape is (H, W, C). These drive the
        # pixel -> normalized canonical conversion (SPEC.md §2.4, §6.4).
        height, width = int(image_bgr.shape[0]), int(image_bgr.shape[1])

        # autodistill BaseModel exposes .predict(input) -> supervision.Detections.
        sv_detections = self._model.predict(image_bgr)

        detections: list[Detection] = []

        # supervision.Detections is iterable / index-addressable; pull the
        # parallel arrays defensively in case confidence/class_id are None.
        xyxy = sv_detections.xyxy
        class_ids = sv_detections.class_id
        confidences = sv_detections.confidence

        n = len(xyxy)
        for i in range(n):
            x1, y1, x2, y2 = (float(v) for v in xyxy[i])

            # Map class_id -> class name. Guard against an out-of-range id
            # (shouldn't happen, but never crash the batch over one bad row).
            label = "unknown"
            if class_ids is not None:
                cid = int(class_ids[i])
                if 0 <= cid < len(self._classes):
                    label = self._classes[cid]
                else:
                    logger.warning(
                        "skipping detection with out-of-range class_id=%d (have %d classes)",
                        cid,
                        len(self._classes),
                    )
                    continue

            score = float(confidences[i]) if confidences is not None else None

            # Convert ABSOLUTE PIXEL xyxy -> canonical normalized xyxy. Wrap the
            # BBox construction: it CLAMPS to [0,1] and RAISES on a degenerate
            # (zero/negative-area) box, which the detector can occasionally emit.
            # Drop such boxes rather than aborting the whole image (SPEC.md §6.4).
            try:
                box = BBox.from_pixel_xyxy(x1, y1, x2, y2, width, height)
            except ValueError as exc:
                logger.debug(
                    "dropping degenerate box %s on %dx%d image: %s",
                    (x1, y1, x2, y2),
                    width,
                    height,
                    exc,
                )
                continue

            detections.append(Detection(label=label, box=box, score=score))

        return detections
