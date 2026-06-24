"""Tiling (SAHI-style slicing) for small/dense fruit in large tree images.

WHY THIS EXISTS (SPEC.md §6.5, §11): in a wide tree shot, distant mandarins can
be only a few pixels wide at full-image downscale and simply vanish before the
detector ever sees them. The fix is to slice the image into overlapping tiles, run
the detector on each tile (where the fruit is now a healthy fraction of the tile),
translate each tile's boxes back into full-image coordinates, and merge duplicate
detections across the seams with NMS.

This module is a clean WRAPPER around any ``Detector`` (SPEC.md §6.4) — it knows
nothing about which backend ran, only that ``base.detect(tile)`` returns CANONICAL
boxes (normalized xyxy, top-left origin) relative to the *tile* it was given. The
coordinate translation is the silent-bug minefield (SPEC.md §12), so every step
is converted through the core converters and commented heavily.

Pure numpy + core only — NO cv2/torch here. ``image_bgr`` is treated as a plain
HxWx[C] numpy array, sliced with ordinary array indexing.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from grove.core import formats
from grove.core.models import BBox, Detection

if TYPE_CHECKING:  # only for type hints; avoids importing numpy/the detector eagerly
    import numpy as np

    from grove.pipeline.detectors.base import Detector

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Tile generation
# -----------------------------------------------------------------------------
def generate_tiles(
    W: int, H: int, tile_size: int, overlap: float
) -> list[tuple[int, int, int, int]]:
    """Tile an ``W x H`` image into overlapping ``tile_size`` squares.

    Returns a list of integer pixel tiles ``(x0, y0, x1, y1)`` with the half-open
    convention ``image[y0:y1, x0:x1]`` (so ``x1 <= W`` and ``y1 <= H`` are valid
    array bounds, never exceeded).

    Coverage formula
    ----------------
    The stride between successive tile origins is::

        step = round(tile_size * (1 - overlap))

    i.e. each tile advances by ``tile_size`` minus the requested overlap. Origins
    are walked at ``0, step, 2*step, ...`` along each axis. To guarantee the
    right/bottom edges are covered (the last full-stride tile usually stops short
    of the edge), the FINAL tile on each axis is CLAMPED so its far corner lands
    exactly on ``W`` / ``H`` and its near corner is pulled back to
    ``max(0, edge - tile_size)``. That last tile may overlap its predecessor by
    more than ``overlap`` — that is intentional and harmless; the seam NMS later
    collapses any resulting duplicates.

    Edge cases:
      * Small image (``W <= tile_size`` and/or that axis fits): a single tile
        spanning the whole axis — never a tile larger than the image.
      * ``step`` is floored to >= 1 so a pathological ``overlap`` near 1.0 cannot
        produce an infinite loop.
      * No zero-area tiles are ever emitted (origins that would duplicate the
        clamped final tile are skipped).
    """
    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive, got {tile_size}")
    if not (0.0 <= overlap < 1.0):
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")

    # Stride: how far each tile origin advances. round() (not floor) so the
    # nominal overlap is honored as closely as integer pixels allow. Guard >= 1
    # so very large overlaps still make forward progress.
    step = max(1, round(tile_size * (1.0 - overlap)))

    xs = _axis_origins(W, tile_size, step)
    ys = _axis_origins(H, tile_size, step)

    tiles: list[tuple[int, int, int, int]] = []
    for y0 in ys:
        # Clamp this tile's far edge to the image; pull the near edge back so the
        # tile keeps its size (up to the image bound). For small axes this yields
        # 0..H. For the last tile it guarantees the bottom edge reaches H.
        y1 = min(y0 + tile_size, H)
        for x0 in xs:
            x1 = min(x0 + tile_size, W)
            # Defensive: never emit a zero/negative-area tile (cannot happen given
            # _axis_origins, but coordinate code earns its paranoia — §12).
            if x1 > x0 and y1 > y0:
                tiles.append((x0, y0, x1, y1))
    return tiles


def _axis_origins(length: int, tile_size: int, step: int) -> list[int]:
    """Tile origins along one axis, with the last origin clamped to cover the edge.

    Walks ``0, step, 2*step, ...`` while the tile would extend into new pixels,
    then ensures a final origin at ``max(0, length - tile_size)`` so the far edge
    is always covered. Deduplicates so the clamped edge tile never duplicates a
    stride tile that already reached the edge.
    """
    # Axis fits in a single tile -> one origin at 0 (tile gets clamped to length).
    if length <= tile_size:
        return [0]

    # Origin of the final tile that exactly reaches the far edge.
    last_origin = length - tile_size

    origins: list[int] = []
    o = 0
    # Advance while the origin still sits strictly before the clamped final
    # origin; once we'd reach or pass it, stop and append last_origin explicitly.
    while o < last_origin:
        origins.append(o)
        o += step
    origins.append(last_origin)

    # The while-loop can land its last appended origin equal to last_origin only
    # if step divides evenly; dedupe defensively to avoid a duplicate edge tile.
    deduped: list[int] = []
    for v in origins:
        if not deduped or deduped[-1] != v:
            deduped.append(v)
    return deduped


# -----------------------------------------------------------------------------
# Cross-seam merge (per-class greedy NMS, in canonical coords)
# -----------------------------------------------------------------------------
def merge_detections(dets: list[Detection], iou_threshold: float) -> list[Detection]:
    """Per-class greedy NMS on CANONICAL boxes — collapses seam duplicates.

    An object straddling a tile boundary is detected once per tile; after the
    boxes are translated to full-image coords they overlap heavily, so we must
    suppress the redundant copies (SPEC.md §6.5: "one box, not two").

    Algorithm (standard greedy NMS, run independently per class):
      1. Group detections by ``label`` so a mandarin never suppresses a different
         class (single class in v1, but keep it correct for multi-class).
      2. Sort each group by ``score`` descending. A ``None`` score (human-drawn,
         or a backend that omits confidence) is treated as LOWEST so real
         detections win the keep-vs-suppress contest.
      3. Greedily keep the top box, drop every remaining box whose IoU with it
         exceeds ``iou_threshold``, repeat on what's left.

    IoU is computed directly on the normalized [0,1] boxes — ratios are
    scale-invariant, so working in normalized units is exact and avoids needing
    W/H here.
    """
    if not dets:
        return []

    # 1) Bucket by class label.
    by_label: dict[str, list[Detection]] = {}
    for d in dets:
        by_label.setdefault(d.label, []).append(d)

    kept: list[Detection] = []
    for label, group in by_label.items():
        # 2) Sort by score desc; None -> -inf so it sorts last (lowest priority).
        ordered = sorted(
            group,
            key=lambda d: (d.score if d.score is not None else float("-inf")),
            reverse=True,
        )
        # 3) Greedy suppression.
        while ordered:
            best = ordered.pop(0)  # highest-scoring remaining box: always kept
            kept.append(best)
            survivors: list[Detection] = []
            for other in ordered:
                if _iou(best.box, other.box) <= iou_threshold:
                    survivors.append(other)  # distinct enough -> keep for next round
                # else: too much overlap with `best` -> suppressed (the duplicate)
            ordered = survivors
    return kept


def _iou(a: BBox, b: BBox) -> float:
    """Intersection-over-union of two canonical (normalized) boxes."""
    # Intersection rectangle: max of the lefts/tops, min of the rights/bottoms.
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)

    iw = ix2 - ix1
    ih = iy2 - iy1
    if iw <= 0.0 or ih <= 0.0:
        return 0.0  # no overlap

    inter = iw * ih
    union = a.area + b.area - inter
    if union <= 0.0:
        return 0.0  # degenerate guard (shouldn't happen: BBox forbids zero area)
    return inter / union


# -----------------------------------------------------------------------------
# TiledDetector — wraps any base Detector with slice -> detect -> translate -> NMS
# -----------------------------------------------------------------------------
class TiledDetector:
    """Run ``base`` over overlapping tiles and merge results into full-image boxes.

    Conforms to the same ``Detector`` contract (``detect(image_bgr) -> list[Detection]``
    with CANONICAL boxes), so callers cannot tell a tiled detector from a plain one
    — exactly the pluggability §6.4 asks for.
    """

    def __init__(self, base: "Detector", tile_size: int, overlap: float, nms_iou: float):
        self.base = base
        self.tile_size = tile_size
        self.overlap = overlap
        self.nms_iou = nms_iou

    def detect(self, image_bgr: "np.ndarray") -> list[Detection]:
        """Tile the image, detect per tile, translate to full-image canonical, merge.

        COORDINATE TRANSLATION (the bug-prone part — §12). For each tile
        ``(x0, y0, x1, y1)`` the base detector returns boxes in the tile's OWN
        canonical frame: normalized to the tile's pixel size ``(tw, th)`` where
        ``tw = x1 - x0``, ``th = y1 - y0``. To lift a tile box into full-image
        canonical we go through pixels in two hops:

          1. tile-canonical -> tile-PIXEL : multiply by the tile size (tw, th)
                 (formats.canonical_to_pixel_xyxy(box, tw, th))
          2. tile-pixel     -> full-PIXEL : add the tile ORIGIN (x0, y0)
          3. full-pixel     -> full-canonical : divide by the full size (W, H)
                 (formats.pixel_xyxy_to_canonical(full_px, W, H))

        Mixing W/H up, or normalizing by the full size before adding the origin,
        are the classic silent failures here — hence the explicit two-hop math.
        """
        # H, W are the full-image pixel dims. numpy shape is (rows=H, cols=W, ...).
        H, W = int(image_bgr.shape[0]), int(image_bgr.shape[1])

        tiles = generate_tiles(W, H, self.tile_size, self.overlap)
        logger.debug("TiledDetector: %d tiles for %dx%d image", len(tiles), W, H)

        all_dets: list[Detection] = []
        for (x0, y0, x1, y1) in tiles:
            tw = x1 - x0  # tile pixel width
            th = y1 - y0  # tile pixel height

            # Slice the tile out of the full image (half-open indexing).
            tile_img = image_bgr[y0:y1, x0:x1]

            # Base detector sees ONLY this tile; its boxes are normalized to it.
            tile_dets = self.base.detect(tile_img)

            for det in tile_dets:
                # Hop 1: tile-canonical -> tile-pixel (de-normalize by tile size).
                px = formats.canonical_to_pixel_xyxy(
                    (det.box.x1, det.box.y1, det.box.x2, det.box.y2), tw, th
                )
                # Hop 2: tile-pixel -> full-image pixel (shift by tile origin).
                full_px = (px[0] + x0, px[1] + y0, px[2] + x0, px[3] + y0)
                # Hop 3: full-pixel -> full-image canonical (normalize by W, H).
                fx1, fy1, fx2, fy2 = formats.pixel_xyxy_to_canonical(full_px, W, H)

                # BBox construction clamps to [0,1] and rejects degenerate boxes;
                # skip any box that collapses after clamping rather than crashing
                # the whole image (mirrors ingest's "skip and log" posture).
                try:
                    box = BBox(x1=fx1, y1=fy1, x2=fx2, y2=fy2)
                except ValueError:
                    logger.debug(
                        "TiledDetector: dropped degenerate translated box %s in tile %s",
                        (fx1, fy1, fx2, fy2),
                        (x0, y0, x1, y1),
                    )
                    continue
                all_dets.append(Detection(label=det.label, box=box, score=det.score))

        # Collapse seam duplicates into single full-image boxes.
        return merge_detections(all_dets, self.nms_iou)
