"""YOLO-World open-vocabulary detector backend (SPEC.md §2.5, §6.4).

This is the *alternate* labeling backend. It exists primarily to PROVE the
``Detector`` abstraction holds: from the outside it is indistinguishable from the
default GroundingDINO backend — same constructor shape, same ``detect()`` contract
(take a BGR image, return CANONICAL ``Detection``s). Keep it thin; all the
backend-specific coordinate quirks must be absorbed here so nothing downstream
ever sees a non-canonical box (§6.4).

Like GroundingDINO this is a "teacher" used to auto-LABEL data, not the robot's
runtime model (§2.1). Only a distilled YOLO *student* is robot-deployable.

GPU note: ``autodistill_yolo_world`` pulls in heavy torch/ultralytics deps that
are not installed on the review-only Mac, so the import is LAZY (inside __init__).
A consumer can import this module to reference the class without those deps; they
are only required when a detector is actually constructed and run on the GPU side.
"""

from __future__ import annotations

import logging

import numpy as np

from grove.core.models import BBox, Detection

logger = logging.getLogger(__name__)


class YOLOWorldDetector:
    """Open-vocabulary detector wrapping ``autodistill_yolo_world.YOLOWorld``.

    Construction maps the ontology (prompt text -> output class name) into an
    autodistill ``CaptionOntology`` and builds the underlying YOLO-World model.
    ``detect()`` adapts its native ``supervision.Detections`` output (absolute
    pixel xyxy) into canonical normalized ``Detection``s.

    NOTE on ``text_threshold``: YOLO-World does not expose a separate text/phrase
    threshold the way GroundingDINO does — it filters on a single box confidence.
    We still accept ``text_threshold`` so every backend shares one uniform
    constructor signature; here it is recorded for provenance but unused.
    """

    def __init__(
        self,
        ontology: dict[str, str],
        box_threshold: float,
        text_threshold: float,
    ) -> None:
        # Lazy heavy imports: only needed when a detector is actually built on the
        # GPU side. Keeping them out of module scope lets the light review/CLI
        # paths import this module on a torch-free Mac.
        from autodistill.detection import CaptionOntology
        from autodistill_yolo_world import YOLOWorld

        self.ontology = ontology
        self.box_threshold = box_threshold
        # Accepted for a uniform constructor across backends; YOLO-World has no
        # distinct text threshold, so this is provenance-only (see class docstring).
        self.text_threshold = text_threshold

        # autodistill CaptionOntology maps PROMPT TEXT -> CLASS NAME. We preserve
        # that exact mapping so the class names written into labels match config.
        self._caption_ontology = CaptionOntology(ontology)

        # Build the underlying open-vocab model from the prompts.
        self._model = YOLOWorld(ontology=self._caption_ontology)

        # Stable ordered class-name list, indexed by class id, to map predicted
        # class ids back to output names. autodistill exposes the ontology's
        # output classes in id order via .classes().
        self._classes: list[str] = list(self._caption_ontology.classes())

    def detect(self, image_bgr: "np.ndarray") -> list[Detection]:
        """Run YOLO-World on one BGR image and return CANONICAL detections.

        The model's native output is a ``supervision.Detections`` with ``xyxy`` in
        ABSOLUTE PIXELS; we divide x by width and y by height (via
        ``BBox.from_pixel_xyxy``) so everything downstream sees only canonical
        normalized boxes (§6.4, §9).
        """
        # image_bgr is (H, W, C) — opencv/numpy convention. Pixel dims drive the
        # pixel->canonical conversion, so read them straight from the array.
        height, width = image_bgr.shape[:2]

        # autodistill backends predict from a file path or an ndarray; pass the
        # array directly. Filter by box confidence at the source.
        sv_detections = self._model.predict(image_bgr, confidence=self.box_threshold)

        # Read supervision's parallel arrays directly (xyxy / class_id / confidence)
        # instead of unpacking the per-row iteration tuple. The tuple's arity has
        # changed across supervision releases (a real version-sensitivity), whereas
        # these attribute arrays are stable — and this mirrors the GroundingDINO
        # backend exactly, so both behave identically from the outside (§6.4).
        xyxy = sv_detections.xyxy
        class_ids = sv_detections.class_id
        confidences = sv_detections.confidence

        detections: list[Detection] = []
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = (float(v) for v in xyxy[i])

            # Guard degenerate boxes BEFORE constructing a BBox: zero/negative
            # span (or a box collapsed to a line) would raise in BBox; skip them
            # rather than crash a folder-scale labeling run.
            if x2 - x1 <= 0 or y2 - y1 <= 0:
                logger.debug("skipping degenerate box from YOLO-World: %s", (x1, y1, x2, y2))
                continue

            try:
                box = BBox.from_pixel_xyxy(x1, y1, x2, y2, width, height)
            except ValueError:
                # A box that collapses to degenerate only after clamping to the
                # image bounds — same treatment: skip, don't crash.
                logger.debug("skipping box degenerate after clamp: %s", (x1, y1, x2, y2))
                continue

            # Map class id -> output class name via the ontology's class list.
            label = "unknown"
            if class_ids is not None:
                cid = int(class_ids[i])
                if 0 <= cid < len(self._classes):
                    label = self._classes[cid]
                else:
                    # Defensive: an out-of-range id means the model/ontology drifted.
                    # Mirror the GroundingDINO backend and DROP the box rather than
                    # silently mislabeling it (keeps backends indistinguishable from
                    # the outside, §6.4; avoids silent mislabels, §12).
                    logger.warning(
                        "skipping detection with out-of-range class_id=%d (have %d classes)",
                        cid,
                        len(self._classes),
                    )
                    continue

            score = float(confidences[i]) if confidences is not None else None
            detections.append(Detection(label=label, box=box, score=score))

        return detections
