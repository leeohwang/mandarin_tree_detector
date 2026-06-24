"""Coordinate-format converters — the single most bug-prone surface in Grove.

CANONICAL FORMAT (the ONE internal representation the entire codebase reasons in):
    normalized **xyxy**, **top-left origin**.
    box = (x1, y1, x2, y2), each value a float in [0, 1].
    Invariants enforced elsewhere (see models.BBox): 0 <= x1 < x2 <= 1, 0 <= y1 < y2 <= 1.

Every external format expresses boxes differently, and the bugs are *silent* — a
mirrored / transposed / off-by-W box looks "almost right" (SPEC.md §12). So we
convert ONLY at the boundaries, using exactly the math from SPEC.md §8, and we
unit-test the round-trips first (tests/test_formats.py).

These are pure functions on plain floats/tuples — NO pydantic, NO numpy, NO import
of models.py (avoids an import cycle and keeps core import-light). W and H are the
image's pixel dimensions.
"""

from __future__ import annotations


# -----------------------------------------------------------------------------
# Canonical  <->  YOLO
#   YOLO box = (cx, cy, w, h): normalized, CENTER-based. W/H are NOT needed
#   because both formats are already normalized to image size.
# -----------------------------------------------------------------------------
def canonical_to_yolo(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Canonical (x1,y1,x2,y2) -> YOLO (cx,cy,w,h). All normalized; SPEC.md §8."""
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0  # center x = midpoint of the two corners
    cy = (y1 + y2) / 2.0  # center y = midpoint of the two corners
    w = x2 - x1           # width  = corner span in x
    h = y2 - y1           # height = corner span in y
    return (cx, cy, w, h)


def yolo_to_canonical(yolo: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """YOLO (cx,cy,w,h) -> canonical (x1,y1,x2,y2). All normalized; SPEC.md §8."""
    cx, cy, w, h = yolo
    x1 = cx - w / 2.0  # left   = center minus half-width
    x2 = cx + w / 2.0  # right  = center plus half-width
    y1 = cy - h / 2.0  # top    = center minus half-height
    y2 = cy + h / 2.0  # bottom = center plus half-height
    return (x1, y1, x2, y2)


# -----------------------------------------------------------------------------
# Canonical  <->  COCO
#   COCO bbox = [x_min, y_min, width, height] in ABSOLUTE PIXELS, top-left origin.
#   Here W/H ARE needed to de-normalize / normalize.
# -----------------------------------------------------------------------------
def canonical_to_coco(
    box: tuple[float, float, float, float], W: int, H: int
) -> tuple[float, float, float, float]:
    """Canonical (x1,y1,x2,y2) -> COCO (x_min,y_min,width,height) in pixels; SPEC.md §8."""
    x1, y1, x2, y2 = box
    x_min = x1 * W                # scale normalized left edge to pixels
    y_min = y1 * H               # scale normalized top edge to pixels
    width = (x2 - x1) * W        # scale normalized width to pixels
    height = (y2 - y1) * H       # scale normalized height to pixels
    return (x_min, y_min, width, height)


def coco_to_canonical(
    coco: tuple[float, float, float, float], W: int, H: int
) -> tuple[float, float, float, float]:
    """COCO (x_min,y_min,width,height) px -> canonical (x1,y1,x2,y2); SPEC.md §8."""
    x_min, y_min, width, height = coco
    x1 = x_min / W                       # left   pixel -> normalized
    y1 = y_min / H                       # top    pixel -> normalized
    x2 = (x_min + width) / W             # right  = (x_min + width) pixel -> normalized
    y2 = (y_min + height) / H            # bottom = (y_min + height) pixel -> normalized
    return (x1, y1, x2, y2)


# -----------------------------------------------------------------------------
# Pixel-xyxy  <->  Canonical
#   Pixel xyxy = absolute-pixel corners (x1,y1,x2,y2). This is what
#   supervision / Grounding DINO emit natively, and what we draw with.
#   Convert by dividing x's by W and y's by H (SPEC.md §6.4, §8).
# -----------------------------------------------------------------------------
def pixel_xyxy_to_canonical(
    xyxy: tuple[float, float, float, float], W: int, H: int
) -> tuple[float, float, float, float]:
    """Absolute-pixel corners -> canonical normalized corners; SPEC.md §6.4/§8."""
    x1, y1, x2, y2 = xyxy
    return (x1 / W, y1 / H, x2 / W, y2 / H)  # x by W, y by H — never swap these


def canonical_to_pixel_xyxy(
    box: tuple[float, float, float, float], W: int, H: int
) -> tuple[float, float, float, float]:
    """Canonical normalized corners -> absolute-pixel corners (for drawing); SPEC.md §8."""
    x1, y1, x2, y2 = box
    return (x1 * W, y1 * H, x2 * W, y2 * H)  # inverse of pixel_xyxy_to_canonical
