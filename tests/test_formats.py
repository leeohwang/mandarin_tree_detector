"""Round-trip + hand-computed tests for the coordinate converters (SPEC.md §8, §13).

Coordinate bugs are SILENT: a mirrored / transposed / off-by-W box looks "almost
right" (SPEC.md §12). The defenses here are:

  1. Round-trip every converter pair and assert identity within 1e-9.
  2. Use NON-SQUARE image sizes (1920x1080 and 640x853) so any W/H swap inside a
     converter produces a wrong answer instead of accidentally cancelling out.
  3. At least one HAND-COMPUTED assertion per converter against a value chosen so
     that a W/H transposition OR an x/y mirror would necessarily FAIL (e.g. a box
     on the RIGHT half of a WIDE image must give COCO x_min == 0.5*W exactly; if
     x and y were swapped, x_min would scale by H instead and the number changes).

These tests are the FIRST line of defense per the spec — they run before the
converters are trusted anywhere downstream.
"""

from __future__ import annotations

import pytest

from grove.core import formats
from grove.core.models import BBox

# Floating-point tolerance for round-trip identity (SPEC.md §8 last paragraph).
TOL = 1e-9

# NON-SQUARE image sizes. 1920x1080 is landscape (W > H); 640x853 is portrait
# (H > W). Using both directions ensures a W/H swap can never silently cancel.
SIZES = [
    (1920, 1080),
    (640, 853),
]

# A spread of canonical boxes (normalized xyxy, top-left origin). Includes:
#   - asymmetric boxes (so cx != cy, w != h) to catch x/y mirroring,
#   - a box touching the (0,0) corner,
#   - a box touching the (1,1) corner,
#   - a box strictly in the right/bottom region (no symmetry to hide a mirror).
CANONICAL_BOXES = [
    (0.10, 0.20, 0.40, 0.50),   # generic asymmetric box, well inside the frame
    (0.00, 0.00, 0.25, 0.10),   # hugs the top-left origin
    (0.60, 0.70, 1.00, 1.00),   # hugs the bottom-right corner
    (0.50, 0.10, 0.90, 0.30),   # right half / upper region (asymmetric in x vs y)
    (0.33, 0.66, 0.77, 0.88),   # arbitrary irrational-ish fractions
]

# Pixel-corner boxes (absolute px) for the pixel<->canonical round-trip. Kept as
# whole/half pixels so the round-trip is exact for the chosen sizes.
PIXEL_XYXY_BOXES = [
    (0.0, 0.0, 100.0, 50.0),
    (192.0, 108.0, 960.0, 540.0),
    (320.0, 200.0, 600.0, 400.0),
]


def _approx(actual, expected, tol=TOL):
    """Assert two coordinate tuples are equal within tol, component-wise."""
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        assert a == pytest.approx(e, abs=tol), f"{actual} != {expected}"


# ---------------------------------------------------------------------------
# Round-trips: canonical -> X -> canonical must be identity.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("box", CANONICAL_BOXES)
def test_canonical_yolo_roundtrip(box):
    """canonical -> YOLO -> canonical == identity (YOLO needs no W/H)."""
    yolo = formats.canonical_to_yolo(box)
    back = formats.yolo_to_canonical(yolo)
    _approx(back, box)


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("box", CANONICAL_BOXES)
def test_canonical_coco_roundtrip(box, size):
    """canonical -> COCO -> canonical == identity on NON-SQUARE sizes."""
    W, H = size
    coco = formats.canonical_to_coco(box, W, H)
    back = formats.coco_to_canonical(coco, W, H)
    _approx(back, box)


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("xyxy", PIXEL_XYXY_BOXES)
def test_pixel_xyxy_roundtrip(xyxy, size):
    """pixel xyxy -> canonical -> pixel xyxy == identity on NON-SQUARE sizes."""
    W, H = size
    canon = formats.pixel_xyxy_to_canonical(xyxy, W, H)
    back = formats.canonical_to_pixel_xyxy(canon, W, H)
    _approx(back, xyxy)


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("box", CANONICAL_BOXES)
def test_canonical_pixel_roundtrip(box, size):
    """canonical -> pixel xyxy -> canonical == identity on NON-SQUARE sizes."""
    W, H = size
    px = formats.canonical_to_pixel_xyxy(box, W, H)
    back = formats.pixel_xyxy_to_canonical(px, W, H)
    _approx(back, box)


# ---------------------------------------------------------------------------
# HAND-COMPUTED assertions — one per converter, each chosen so that a W/H
# transposition OR an x/y mirror would change the number and FAIL the test.
# ---------------------------------------------------------------------------
def test_canonical_to_yolo_handcomputed():
    """Box (0.2,0.4, 0.6,0.8): cx=0.4 cy=0.6 w=0.4 h=0.4.

    cx != cy here, so a swap of (x,y) would flip 0.4<->0.6 and fail.
    """
    cx, cy, w, h = formats.canonical_to_yolo((0.2, 0.4, 0.6, 0.8))
    assert cx == pytest.approx(0.4, abs=TOL)
    assert cy == pytest.approx(0.6, abs=TOL)
    assert w == pytest.approx(0.4, abs=TOL)
    assert h == pytest.approx(0.4, abs=TOL)


def test_yolo_to_canonical_handcomputed():
    """YOLO (0.5,0.25, 0.2,0.1) -> canonical (0.4,0.2, 0.6,0.3).

    cx != cy and w != h, so any x/y mirror would corrupt the corners.
    """
    x1, y1, x2, y2 = formats.yolo_to_canonical((0.5, 0.25, 0.2, 0.1))
    assert x1 == pytest.approx(0.4, abs=TOL)
    assert y1 == pytest.approx(0.2, abs=TOL)
    assert x2 == pytest.approx(0.6, abs=TOL)
    assert y2 == pytest.approx(0.3, abs=TOL)


def test_canonical_to_coco_handcomputed_wide():
    """Right-half box on a WIDE 1920x1080 image -> x_min == 0.5*W exactly.

    box = (0.5, 0.0, 1.0, 0.5).
    x_min  = 0.5 * 1920 = 960   (NOT 0.5*1080=540, which a W/H swap would give)
    y_min  = 0.0 * 1080 = 0
    width  = 0.5 * 1920 = 960
    height = 0.5 * 1080 = 540
    The x_min == 960 (not 540) assertion is exactly what fails on a W/H swap;
    width != height (960 != 540) is what fails on an x/y mirror.
    """
    W, H = 1920, 1080
    x_min, y_min, width, height = formats.canonical_to_coco((0.5, 0.0, 1.0, 0.5), W, H)
    assert x_min == pytest.approx(960.0, abs=TOL)   # 0.5 * W
    assert y_min == pytest.approx(0.0, abs=TOL)
    assert width == pytest.approx(960.0, abs=TOL)   # 0.5 * W
    assert height == pytest.approx(540.0, abs=TOL)  # 0.5 * H  (!= width -> catches mirror)


def test_coco_to_canonical_handcomputed_wide():
    """COCO [960,0, 960,540] on 1920x1080 -> canonical (0.5,0.0, 1.0,0.5).

    x1 = 960/1920 = 0.5 ; if W/H were swapped, 960/1080 = 0.888.. and fails.
    """
    W, H = 1920, 1080
    x1, y1, x2, y2 = formats.coco_to_canonical((960.0, 0.0, 960.0, 540.0), W, H)
    assert x1 == pytest.approx(0.5, abs=TOL)
    assert y1 == pytest.approx(0.0, abs=TOL)
    assert x2 == pytest.approx(1.0, abs=TOL)
    assert y2 == pytest.approx(0.5, abs=TOL)


def test_pixel_xyxy_to_canonical_handcomputed_portrait():
    """Pixel corners on a PORTRAIT 640x853 image.

    xyxy = (320, 0, 640, 853) -> (0.5, 0.0, 1.0, 1.0).
    x1 = 320/640 = 0.5 ; a W/H swap (320/853=0.375) would fail.
    y2 = 853/853 = 1.0 ; a W/H swap (853/640>1) would fail.
    """
    W, H = 640, 853
    x1, y1, x2, y2 = formats.pixel_xyxy_to_canonical((320.0, 0.0, 640.0, 853.0), W, H)
    assert x1 == pytest.approx(0.5, abs=TOL)
    assert y1 == pytest.approx(0.0, abs=TOL)
    assert x2 == pytest.approx(1.0, abs=TOL)
    assert y2 == pytest.approx(1.0, abs=TOL)


def test_canonical_to_pixel_xyxy_handcomputed_portrait():
    """Canonical (0.5,0.0, 1.0,1.0) on 640x853 -> pixels (320,0, 640,853).

    x1 = 0.5*640 = 320 (not 0.5*853=426.5, the W/H-swap value).
    """
    W, H = 640, 853
    x1, y1, x2, y2 = formats.canonical_to_pixel_xyxy((0.5, 0.0, 1.0, 1.0), W, H)
    assert x1 == pytest.approx(320.0, abs=TOL)
    assert y1 == pytest.approx(0.0, abs=TOL)
    assert x2 == pytest.approx(640.0, abs=TOL)
    assert y2 == pytest.approx(853.0, abs=TOL)


# ---------------------------------------------------------------------------
# Model-layer round-trips: BBox.to_*/from_* delegate to formats.py, but verify
# the wiring (argument order, W/H placement) is correct end-to-end.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("box", CANONICAL_BOXES)
def test_bbox_yolo_roundtrip(box):
    """BBox -> to_yolo -> from_yolo -> BBox preserves all four corners."""
    bb = BBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3])
    cx, cy, w, h = bb.to_yolo()
    back = BBox.from_yolo(cx, cy, w, h)
    assert back.x1 == pytest.approx(bb.x1, abs=TOL)
    assert back.y1 == pytest.approx(bb.y1, abs=TOL)
    assert back.x2 == pytest.approx(bb.x2, abs=TOL)
    assert back.y2 == pytest.approx(bb.y2, abs=TOL)


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("box", CANONICAL_BOXES)
def test_bbox_coco_roundtrip(box, size):
    """BBox -> to_coco(W,H) -> from_coco(...,W,H) -> BBox on NON-SQUARE sizes."""
    W, H = size
    bb = BBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3])
    x_min, y_min, w, h = bb.to_coco(W, H)
    back = BBox.from_coco(x_min, y_min, w, h, W, H)
    assert back.x1 == pytest.approx(bb.x1, abs=TOL)
    assert back.y1 == pytest.approx(bb.y1, abs=TOL)
    assert back.x2 == pytest.approx(bb.x2, abs=TOL)
    assert back.y2 == pytest.approx(bb.y2, abs=TOL)


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("box", CANONICAL_BOXES)
def test_bbox_pixel_xyxy_roundtrip(box, size):
    """BBox -> to_pixel_xyxy(W,H) -> from_pixel_xyxy(...,W,H) -> BBox."""
    W, H = size
    bb = BBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3])
    px = bb.to_pixel_xyxy(W, H)
    back = BBox.from_pixel_xyxy(px[0], px[1], px[2], px[3], W, H)
    assert back.x1 == pytest.approx(bb.x1, abs=TOL)
    assert back.y1 == pytest.approx(bb.y1, abs=TOL)
    assert back.x2 == pytest.approx(bb.x2, abs=TOL)
    assert back.y2 == pytest.approx(bb.y2, abs=TOL)


def test_bbox_coco_handcomputed_wide():
    """End-to-end model check: right-half box on 1920x1080 -> COCO x_min=960.

    Catches a W/H wiring mistake in BBox.to_coco specifically (e.g. passing H,W).
    """
    bb = BBox(x1=0.5, y1=0.0, x2=1.0, y2=0.5)
    x_min, y_min, w, h = bb.to_coco(1920, 1080)
    assert x_min == pytest.approx(960.0, abs=TOL)
    assert y_min == pytest.approx(0.0, abs=TOL)
    assert w == pytest.approx(960.0, abs=TOL)
    assert h == pytest.approx(540.0, abs=TOL)
