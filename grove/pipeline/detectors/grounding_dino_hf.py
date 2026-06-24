"""HuggingFace `transformers` Grounding DINO backend (SPEC.md §2.1, §2.5, §6.4).

================================  READ THIS  ================================
This is the SAME Grounding DINO model as the autodistill backend
(detectors/grounding_dino.py) — the IDEA-Research open-vocabulary detector —
but loaded through HuggingFace `transformers` instead of `autodistill` +
`groundingdino-py`. It exists so the pipeline can run the teacher/auto-labeler
on a machine where the autodistill stack is impractical: the target dev box is
an Intel Mac with NO CUDA GPU, and `autodistill-grounding-dino` pulls in
`groundingdino-py`, which compiles native CUDA/C++ ops. The `transformers`
implementation is pure-Python, installs as plain wheels, and runs on CPU (a
single large image is ~seconds-to-tens-of-seconds — fine for one-off labeling).

Still a TEACHER, never the robot's runtime. Like every backend here it only
DRAFTS labels for the mandatory human review step; the deployable real-time
model is the distilled YOLO student (pipeline/train.py). (SPEC.md §2.1, §12)
============================================================================

From the outside it satisfies the Detector protocol (detectors/base.py): take a
BGR image, return a list[Detection] whose boxes are already CANONICAL (normalized
xyxy, top-left origin, in [0, 1]). Grounding DINO's native output is ABSOLUTE
PIXEL xyxy; that quirk is normalized away here so nothing downstream sees
backend-specific coordinates (SPEC.md §2.4, §6.4).

Heavy imports (torch / transformers) are LAZY: they happen inside __init__, so a
light consumer (e.g. the CLI on a review-only Mac install) can import this module
without those packages present. Only instantiating the detector pulls them in.

Knobs via environment (no config change needed):
  * GROVE_GDINO_HF_MODEL  — HF model id (default "IDEA-Research/grounding-dino-tiny").
                            Use "IDEA-Research/grounding-dino-base" for higher
                            accuracy at more compute.
"""

from __future__ import annotations

import inspect
import logging
import os
from typing import TYPE_CHECKING

from grove.core.models import BBox, Detection
from grove.pipeline.tiling import merge_detections

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    import numpy as np

logger = logging.getLogger(__name__)

# Default checkpoint. "tiny" is plenty for large, salient objects like whole
# trees and is markedly faster on CPU than "base"; override via env if needed.
_DEFAULT_MODEL = "IDEA-Research/grounding-dino-tiny"

# Default IoU for de-duplicating overlapping boxes. Grounding DINO frequently
# emits several near-identical boxes for the SAME object (e.g. one tree getting
# 3 stacked boxes), which is wrong for per-instance labels. A greedy NMS pass
# collapses them so each object becomes one box. 0.0 disables it.
_DEFAULT_NMS_IOU = 0.5


class HFGroundingDINODetector:
    """Open-vocabulary Grounding DINO via HuggingFace `transformers`.

    TEACHER MODEL — auto-labeling only, never the robot runtime (see module
    docstring). Construct with an ontology mapping prompt text -> class name plus
    the box/text confidence thresholds, then call .detect(image_bgr).
    """

    def __init__(
        self,
        ontology: dict[str, str],
        box_threshold: float,
        text_threshold: float,
    ) -> None:
        # --- LAZY heavy imports -------------------------------------------------
        # torch + transformers are heavy and absent on a review-only Mac. Import
        # them here (not at module top) so this file stays importable there.
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self._torch = torch
        self._box_threshold = float(box_threshold)
        self._text_threshold = float(text_threshold)

        # --- Build the text prompt + phrase -> class-name map -------------------
        # Grounding DINO is conditioned on a lowercase, period-delimited caption.
        # The ontology maps PROMPT TEXT -> CLASS NAME: the prompt is what the model
        # sees, the class name is what gets written into the exported labels. We
        # keep an ordered, de-duplicated class list (index == YOLO class id order)
        # and a phrase->class lookup to translate each matched phrase back to its
        # class name.
        self._classes: list[str] = list(dict.fromkeys(ontology.values()))
        self._phrase_to_class: dict[str, str] = {}
        phrases: list[str] = []
        for prompt_text, class_name in ontology.items():
            phrase = prompt_text.strip().lower().rstrip(" .")
            if not phrase:
                continue
            phrases.append(phrase)
            self._phrase_to_class[phrase] = class_name
        if not phrases:
            raise ValueError("ontology produced an empty Grounding DINO prompt")
        # Canonical Grounding DINO caption format: "phrase a . phrase b ."
        self._text_prompt = " . ".join(phrases) + " ."

        # --- Load model + processor -------------------------------------------
        # Source resolution (see _resolve_model_source): explicit env override,
        # else a locally-cached copy if present, else the HF hub id (downloads on
        # first use). Preferring the local copy lets `grove detect` run OFFLINE on
        # the dev Mac, whose TLS-intercepting proxy breaks the HF metadata HEAD.
        model_id = self._resolve_model_source()
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        self._model.eval()
        # Use a GPU if one is present (e.g. Kaggle), else CPU (the Mac dev box).
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)

        # NMS IoU for collapsing Grounding DINO's overlapping duplicate boxes so
        # each object yields one box (env-tunable; set to 0 to disable).
        self._nms_iou = float(os.environ.get("GROVE_GDINO_HF_NMS_IOU", _DEFAULT_NMS_IOU))

        logger.info(
            "HFGroundingDINODetector ready (TEACHER/labeler, not robot runtime): "
            "model=%s device=%s prompt=%r classes=%s box_threshold=%.3f text_threshold=%.3f nms_iou=%.2f",
            model_id,
            self._device,
            self._text_prompt,
            self._classes,
            self._box_threshold,
            self._text_threshold,
            self._nms_iou,
        )

    @staticmethod
    def _resolve_model_source() -> str:
        """Pick where to load Grounding DINO weights from.

        Priority:
          1. ``GROVE_GDINO_HF_MODEL`` env var (explicit override: an HF id or a
             local directory).
          2. A locally-cached copy at ``~/.cache/grove_models/<repo>`` if it holds
             a ``config.json``. This is what the project's weight-fetch step writes
             so detection runs OFFLINE — essential on the dev Mac, whose proxy
             breaks the Hub's metadata HEAD request.
          3. The default HF hub id (downloads on first use where the Hub is
             reachable, e.g. Kaggle).
        """
        from pathlib import Path

        env = os.environ.get("GROVE_GDINO_HF_MODEL")
        if env:
            return env
        repo_name = _DEFAULT_MODEL.split("/")[-1]
        local = Path.home() / ".cache" / "grove_models" / repo_name
        if (local / "config.json").is_file():
            return str(local)
        return _DEFAULT_MODEL

    def detect(self, image_bgr: "np.ndarray") -> list[Detection]:
        """Run Grounding DINO on one image and return canonical Detections.

        `image_bgr` is an OpenCV-style HxWx3 ndarray in BGR order (the project's
        in-memory image convention). transformers' image processor expects RGB,
        so we flip channels here (the ONE place that knows about BGR vs RGB).
        Post-processing yields ABSOLUTE PIXEL xyxy boxes, which we convert to
        canonical via BBox.from_pixel_xyxy using the image's pixel W/H.
        """
        from PIL import Image

        torch = self._torch
        height, width = int(image_bgr.shape[0]), int(image_bgr.shape[1])

        # BGR -> RGB for transformers. `[:, :, ::-1]` is a view; PIL needs a real
        # array, so copy via Image.fromarray on a contiguous RGB array.
        image_rgb = image_bgr[:, :, ::-1]
        pil_image = Image.fromarray(image_rgb.copy())

        inputs = self._processor(
            images=pil_image, text=self._text_prompt, return_tensors="pt"
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        # target_sizes is (height, width) per the transformers convention; this is
        # what scales the boxes back to absolute pixels of the ORIGINAL image.
        results = self._post_process(outputs, inputs, (height, width))
        result = results[0]

        boxes = result["boxes"]
        scores = result["scores"]
        # Label key was renamed "labels" -> "text_labels" across transformers
        # versions; accept either. Values are the matched phrase strings.
        labels = result.get("text_labels", result.get("labels"))

        detections: list[Detection] = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = (float(v) for v in boxes[i].tolist())
            score = float(scores[i])

            raw_label = labels[i] if labels is not None else ""
            if not isinstance(raw_label, str):
                raw_label = ""
            label = self._map_label(raw_label)

            # ABSOLUTE PIXEL xyxy -> canonical. BBox clamps to [0,1] and RAISES on a
            # degenerate (zero/negative-area) box; drop those rather than abort the
            # image (SPEC.md §6.4).
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

        # Collapse overlapping duplicate boxes (Grounding DINO emits several near-
        # identical boxes per object) so each object is a single box. Reuses the
        # same canonical greedy NMS the tiling stage uses for seam merges.
        if self._nms_iou > 0.0 and len(detections) > 1:
            before = len(detections)
            detections = merge_detections(detections, self._nms_iou)
            if len(detections) != before:
                logger.info(
                    "NMS(iou=%.2f) collapsed %d -> %d box(es)",
                    self._nms_iou,
                    before,
                    len(detections),
                )

        return detections

    def _post_process(self, outputs, inputs, target_size: tuple[int, int]):
        """Call processor.post_process_grounded_object_detection robustly.

        Its signature has churned across transformers releases (``box_threshold``
        -> ``threshold``; ``input_ids`` sometimes required, sometimes not). We
        introspect the actual signature and pass only the kwargs it accepts, so
        this backend keeps working across versions without pinning.
        """
        proc = self._processor
        fn = proc.post_process_grounded_object_detection
        params = inspect.signature(fn).parameters

        kwargs: dict = {"target_sizes": [target_size]}
        if "input_ids" in params:
            kwargs["input_ids"] = inputs["input_ids"]
        # Box-confidence threshold: newer transformers renamed the kwarg.
        if "box_threshold" in params:
            kwargs["box_threshold"] = self._box_threshold
        elif "threshold" in params:
            kwargs["threshold"] = self._box_threshold
        if "text_threshold" in params:
            kwargs["text_threshold"] = self._text_threshold

        return fn(outputs, **kwargs)

    def _map_label(self, raw_label: str) -> str:
        """Translate a detected phrase back to its configured class name.

        Grounding DINO returns the matched caption fragment (e.g. "tree" or a
        partial token span). We match it against the ontology phrases tolerantly;
        an unrecognized fragment falls back to the first class so a detection is
        never silently dropped over a labeling mismatch.
        """
        fallback = self._classes[0] if self._classes else "object"
        if not raw_label:
            return fallback
        text = raw_label.strip().lower().rstrip(" .")
        if text in self._phrase_to_class:
            return self._phrase_to_class[text]
        # Substring match either direction (handles partial caption spans).
        for phrase, class_name in self._phrase_to_class.items():
            if phrase and (phrase in text or text in phrase):
                return class_name
        # Word-overlap fallback (e.g. "tree" within "a tall tree").
        tokens = set(text.split())
        for phrase, class_name in self._phrase_to_class.items():
            if tokens & set(phrase.split()):
                return class_name
        return fallback
