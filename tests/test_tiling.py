"""Tiling tests: coverage, coordinate TRANSLATION, and seam MERGE (SPEC.md §6.5, §13).

Tiling is the silent-coordinate-bug minefield (§12): a box can come back mirrored,
off by the tile origin, or normalized against the wrong dimension and still look
"almost right". These tests pin the three things §13 demands:

  (a) generate_tiles covers the WHOLE image including the right/bottom edges,
      never exceeds the array bounds, handles a sub-tile image (single tile) and
      an exact multiple cleanly;
  (b) a fixed tile-local box is translated to the correct FULL-image canonical
      location (hand-computed against a known tile);
  (c) an object straddling a tile boundary, detected once in each adjacent tile,
      collapses to ONE box after NMS — not two.

NON-SQUARE images throughout, so any W/H transposition fails instead of cancelling.
NO cv2/torch — images are plain numpy arrays from the make_array_image fixture.
"""

from __future__ import annotations

import numpy as np
import pytest

from grove.core.models import BBox, Detection
from grove.pipeline import tiling
from grove.pipeline.tiling import TiledDetector, generate_tiles, merge_detections


# ---------------------------------------------------------------------------
# Small custom detectors for the geometry tests (the shared StubDetector returns
# the SAME box for every tile, which is right for the merge test but not for a
# clean single-tile translation check).
# ---------------------------------------------------------------------------
class FixedTileBoxDetector:
    """Returns ONE fixed tile-local canonical box on every detect() call."""

    def __init__(self, box: BBox, label: str = "mandarin", score: float = 0.8):
        self._det = Detection(label=label, box=box, score=score)

    def detect(self, image_bgr) -> list[Detection]:  # noqa: ANN001
        return [self._det.model_copy(deep=True)]


# ===========================================================================
# (a) generate_tiles coverage / bounds
# ===========================================================================
def _covers_every_pixel(tiles, W, H) -> bool:
    """True iff the union of tiles covers every pixel of the WxH grid."""
    covered = np.zeros((H, W), dtype=bool)
    for (x0, y0, x1, y1) in tiles:
        covered[y0:y1, x0:x1] = True
    return bool(covered.all())


@pytest.mark.parametrize(
    "W,H,tile_size,overlap",
    [
        (1000, 600, 400, 0.0),    # non-square, no overlap, non-multiple
        (1000, 600, 400, 0.20),   # non-square, with overlap
        (1280, 720, 640, 0.20),   # 720p, 640 tiles (the config default tile_size)
        (1201, 799, 300, 0.10),   # awkward primes-ish dims to stress edge clamping
    ],
)
def test_generate_tiles_covers_and_stays_in_bounds(W, H, tile_size, overlap):
    tiles = generate_tiles(W, H, tile_size, overlap)
    assert tiles, "expected at least one tile"
    for (x0, y0, x1, y1) in tiles:
        # Never exceed the array bounds (half-open: x1<=W, y1<=H are valid slices).
        assert 0 <= x0 < x1 <= W
        assert 0 <= y0 < y1 <= H
        # Never produce a tile bigger than the requested size on either axis.
        assert x1 - x0 <= tile_size
        assert y1 - y0 <= tile_size
    # The whole image must be covered, right/bottom edges included.
    assert _covers_every_pixel(tiles, W, H)


def test_generate_tiles_small_image_single_tile():
    """An image smaller than tile_size on both axes -> exactly one full tile."""
    W, H = 300, 200  # both < tile_size=640; non-square
    tiles = generate_tiles(W, H, tile_size=640, overlap=0.20)
    assert tiles == [(0, 0, W, H)]


def test_generate_tiles_exact_multiple_no_overlap():
    """Exact multiples with zero overlap tile edge-to-edge, fully covering."""
    W, H = 1200, 800  # 3x2 grid of 400-px tiles, non-square
    tiles = generate_tiles(W, H, tile_size=400, overlap=0.0)
    assert _covers_every_pixel(tiles, W, H)
    # Far edges reached exactly.
    assert max(x1 for (_, _, x1, _) in tiles) == W
    assert max(y1 for (_, _, _, y1) in tiles) == H
    # No tile exceeds bounds; with exact tiling each is full-size.
    for (x0, y0, x1, y1) in tiles:
        assert (x1 - x0, y1 - y0) == (400, 400)


# ===========================================================================
# (b) coordinate TRANSLATION: tile-local box -> full-image canonical location
# ===========================================================================
def test_translation_single_tile_identity(make_array_image):
    """When the image IS one tile, a tile-local box maps to the same canonical box.

    Image 300x200 (< tile_size) => single tile spanning the whole image, so the
    tile frame == the full frame and translation is the identity.
    """
    img = make_array_image(width=300, height=200)
    tile_box = BBox(x1=0.2, y1=0.4, x2=0.6, y2=0.8)
    det = TiledDetector(
        FixedTileBoxDetector(tile_box), tile_size=640, overlap=0.20, nms_iou=0.5
    )
    out = det.detect(img)
    assert len(out) == 1
    b = out[0].box
    assert (b.x1, b.y1, b.x2, b.y2) == pytest.approx((0.2, 0.4, 0.6, 0.8), abs=1e-9)


def test_translation_known_tile_handcomputed():
    """Hand-checked translation for a SPECIFIC interior tile on a non-square image.

    Setup: W=1000, H=600, tile_size=400, overlap=0.0.
      x-origins -> [0, 400, 600]  (last clamped to W-tile_size=600)
      y-origins -> [0, 200]       (last clamped to H-tile_size=200)
    Pick the tile (x0,y0,x1,y1) = (400, 200, 800, 600): a 400x400 tile.

    A tile-local canonical box (0.25, 0.5, 0.75, 1.0) means, in tile PIXELS:
      x: 0.25*400=100 .. 0.75*400=300   ->  +x0=400  -> full px 500 .. 700
      y: 0.5 *400=200 .. 1.0 *400=400   ->  +y0=200  -> full px 400 .. 600
    Full-image canonical = (500/1000, 400/600, 700/1000, 600/600)
                         = (0.5, 0.6666.., 0.7, 1.0).

    A W/H swap (normalizing the y's by W=1000 instead of H=600) would give
    400/1000=0.4 not 0.6667 -> this assertion catches it.
    """
    W, H = 1000, 600
    target_tile = (400, 200, 800, 600)

    # A detector that emits the tile-local box ONLY for the target tile (matched
    # by the sliced array's shape), and nothing for any other tile — so exactly
    # one translated box comes out and we can assert it precisely.
    tw = target_tile[2] - target_tile[0]
    th = target_tile[3] - target_tile[1]
    local = BBox(x1=0.25, y1=0.5, x2=0.75, y2=1.0)

    class OnlyTargetTile:
        def detect(self, image_bgr):  # noqa: ANN001
            h, w = image_bgr.shape[0], image_bgr.shape[1]
            if (w, h) == (tw, th):
                return [Detection(label="mandarin", box=local.model_copy(deep=True), score=0.7)]
            return []

    # Sanity: the target tile is actually generated.
    assert target_tile in generate_tiles(W, H, 400, 0.0)

    img = np.zeros((H, W, 3), dtype=np.uint8)
    det = TiledDetector(OnlyTargetTile(), tile_size=400, overlap=0.0, nms_iou=0.5)
    out = det.detect(img)

    # Several tiles share the 400x400 shape (it's an interior/edge size), so the
    # detector fires for each; they all map to the SAME local fraction of their
    # OWN tile -> distinct full-image boxes. Assert our hand-computed one is present.
    expected = (0.5, 400.0 / 600.0, 0.7, 1.0)
    matches = [
        b for b in (d.box for d in out)
        if (b.x1, b.y1, b.x2, b.y2) == pytest.approx(expected, abs=1e-9)
    ]
    assert matches, f"expected translated box {expected} not found in {[ (d.box.x1,d.box.y1,d.box.x2,d.box.y2) for d in out]}"


# ===========================================================================
# (c) SEAM MERGE: a boundary-straddling object collapses to ONE box, not two.
# ===========================================================================
def test_merge_collapses_seam_duplicates_direct():
    """merge_detections: two heavily-overlapping same-class boxes -> one kept."""
    a = Detection(label="mandarin", box=BBox(x1=0.40, y1=0.40, x2=0.60, y2=0.60), score=0.9)
    b = Detection(label="mandarin", box=BBox(x1=0.41, y1=0.41, x2=0.61, y2=0.61), score=0.8)
    kept = merge_detections([a, b], iou_threshold=0.5)
    assert len(kept) == 1
    # The higher-scoring box wins.
    assert kept[0].score == pytest.approx(0.9)


def test_merge_keeps_distinct_boxes():
    """Two far-apart boxes (IoU 0) survive — NMS must not over-suppress."""
    a = Detection(label="mandarin", box=BBox(x1=0.05, y1=0.05, x2=0.15, y2=0.15), score=0.9)
    b = Detection(label="mandarin", box=BBox(x1=0.80, y1=0.80, x2=0.95, y2=0.95), score=0.8)
    kept = merge_detections([a, b], iou_threshold=0.5)
    assert len(kept) == 2


def test_merge_per_class_independent():
    """Overlapping boxes of DIFFERENT classes are not merged into one."""
    a = Detection(label="mandarin", box=BBox(x1=0.40, y1=0.40, x2=0.60, y2=0.60), score=0.9)
    b = Detection(label="leaf", box=BBox(x1=0.41, y1=0.41, x2=0.61, y2=0.61), score=0.8)
    kept = merge_detections([a, b], iou_threshold=0.5)
    assert len(kept) == 2
    assert {d.label for d in kept} == {"mandarin", "leaf"}


def test_seam_merge_through_tiled_detector():
    """End-to-end: one physical fruit on a tile seam -> ONE box after TiledDetector.

    A fruit centered exactly on a horizontal+vertical seam appears in BOTH the
    left and right (and possibly top/bottom) tiles. With overlap, each tile sees
    (part of) it; their translated full-image boxes overlap heavily and NMS must
    collapse them to a single detection (§6.5: "one box, not two").

    We model this with a detector that, given a tile, returns a box positioned so
    that across adjacent overlapping tiles the translated boxes land on (nearly)
    the SAME full-image region. The simplest faithful model: every tile reports a
    box covering its FULL extent of the overlap zone won't generalize, so instead
    we place a single fixed FULL-IMAGE-anchored object and let the StubDetector
    report it per tile via a content-aware stub.
    """
    W, H = 1000, 600  # non-square
    tile_size, overlap = 400, 0.30

    tiles = generate_tiles(W, H, tile_size, overlap)
    # The physical object, in full-image PIXELS, straddling tile boundaries.
    obj_px = (460.0, 280.0, 540.0, 360.0)  # ~80x80 fruit near the image center

    class SeamObjectDetector:
        """Reports the object IF it intersects the current tile, expressed in the
        tile's own canonical frame (what a real per-tile detector would do)."""

        def __init__(self):
            self._tiles = tiles

        def detect(self, image_bgr):  # noqa: ANN001
            th, tw = image_bgr.shape[0], image_bgr.shape[1]
            # Recover which tile this is by matching size+content is fragile; we
            # instead infer the origin from a 1-pixel marker we stamp below. To
            # keep it simple and robust, we match against the unique tile whose
            # (w,h) and a stamped origin marker correspond. See stamping below.
            ox = int(image_bgr[0, 0, 0]) | (int(image_bgr[0, 0, 1]) << 8)
            oy = int(image_bgr[0, 1, 0]) | (int(image_bgr[0, 1, 1]) << 8)
            x0, y0, x1, y1 = ox, oy, ox + tw, oy + th
            # Intersect object with this tile (pixels).
            ix0 = max(obj_px[0], x0)
            iy0 = max(obj_px[1], y0)
            ix1 = min(obj_px[2], x1)
            iy1 = min(obj_px[3], y1)
            if ix1 <= ix0 or iy1 <= iy0:
                return []
            # Express the intersection in this tile's canonical frame.
            bx = BBox(
                x1=(ix0 - x0) / tw,
                y1=(iy0 - y0) / th,
                x2=(ix1 - x0) / tw,
                y2=(iy1 - y0) / th,
            )
            return [Detection(label="mandarin", box=bx, score=0.85)]

    # Build the full image and stamp each tile's origin into its top-left pixels so
    # the stub can recover (x0,y0). We must stamp the FULL image such that, after
    # TiledDetector slices tile (x0,y0,..), pixel (0,0) of the slice encodes x0 and
    # pixel (0,1) encodes y0. Slice pixel (0,0) is full-image pixel (y0, x0).
    img = np.zeros((H, W, 3), dtype=np.uint8)
    for (x0, y0, x1, y1) in tiles:
        img[y0, x0, 0] = x0 & 0xFF
        img[y0, x0, 1] = (x0 >> 8) & 0xFF
        img[y0, x0 + 1, 0] = y0 & 0xFF
        img[y0, x0 + 1, 1] = (y0 >> 8) & 0xFF

    det = TiledDetector(SeamObjectDetector(), tile_size=tile_size, overlap=overlap, nms_iou=0.5)
    out = det.detect(img)

    # The single physical fruit must come back as exactly ONE merged box.
    assert len(out) == 1, f"seam object should collapse to one box, got {len(out)}"
    b = out[0].box
    # And it should sit at the object's true full-image canonical location.
    exp = (460.0 / W, 280.0 / H, 540.0 / W, 360.0 / H)
    assert (b.x1, b.y1, b.x2, b.y2) == pytest.approx(exp, abs=1e-6)
