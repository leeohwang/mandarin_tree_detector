"""Export INTEGRATION test — the REAL deliverable is the coordinate files (SPEC.md §1, §6.7, §13).

§13 asks for an integration test that takes a small hand-built Manifest through
write_dataset and proves a STRUCTURALLY VALID YOLO + COCO dataset comes out. This
is where the project's central reframe lives: we assert the *coordinate files* are
correct (values in range, class ids valid, COCO pixels in bounds), not that any
image looks pretty.

Specifically pinned (the task brief + §6.7/§16):
  * the §6.7 layout exists (data.yaml, images/{train,val}, labels/{train,val},
    annotations/instances_{train,val}.json);
  * YOLO label values are all in [0,1] and class ids are in range;
  * COCO JSON is valid, bbox is ABSOLUTE PIXELS and within image bounds;
  * a 'skipped' image is EXCLUDED entirely;
  * a zero-box image yields an EMPTY label file (a valid negative example);
  * the train/val split is DETERMINISTIC across two runs with the same seed.

Non-square image sizes throughout. Pillow writes the prepared image files that
export copies; NO cv2/torch.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

import grove
from grove.core.models import BBox, Detection, ImageRecord, Manifest
from grove.pipeline.export import write_dataset


CLASSES = ["mandarin"]


def _build_manifest_and_files(work_dir: Path) -> Manifest:
    """Create prepared image files on disk + a Manifest describing them.

    Three images, all non-square:
      * img_a (800x600): two mandarin boxes -> non-empty label file.
      * img_b (640x853): ZERO boxes -> must yield an EMPTY label file.
      * img_c (500x900): SKIPPED -> must be excluded from the dataset entirely.
    """
    images_dir = work_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        ("img_a", "img_a.jpg", 800, 600),
        ("img_b", "img_b.jpg", 640, 853),
        ("img_c", "img_c.jpg", 500, 900),
    ]
    for _id, fname, w, h in specs:
        Image.new("RGB", (w, h), (90, 120, 60)).save(images_dir / fname)

    records = [
        ImageRecord(
            id="img_a",
            path="images/img_a.jpg",
            width=800,
            height=600,
            status="pending",
            detections=[
                # Right-half box: on COCO this MUST start at x_min=0.5*800=400 px,
                # which only holds if W (not H) scales x — catches a W/H swap.
                Detection(label="mandarin", box=BBox(x1=0.5, y1=0.1, x2=0.9, y2=0.4), score=0.88),
                Detection(label="mandarin", box=BBox(x1=0.05, y1=0.6, x2=0.25, y2=0.95), score=0.71),
            ],
        ),
        ImageRecord(
            id="img_b",
            path="images/img_b.jpg",
            width=640,
            height=853,
            status="reviewed",
            detections=[],  # zero boxes -> empty label file
        ),
        ImageRecord(
            id="img_c",
            path="images/img_c.jpg",
            width=500,
            height=900,
            status="skipped",  # must be excluded
            detections=[
                Detection(label="mandarin", box=BBox(x1=0.1, y1=0.1, x2=0.2, y2=0.2), score=0.5),
            ],
        ),
    ]

    return Manifest.new(
        records,
        detector_backend="grounding_dino",
        ontology={"mandarin orange on tree": "mandarin"},
        box_threshold=0.30,
        text_threshold=0.25,
        tiling=False,
        grove_version=grove.__version__,
    )


def _all_split_files(export_root: Path, subdir: str) -> dict[str, list[str]]:
    return {
        "train": sorted(p.name for p in (export_root / subdir / "train").iterdir()),
        "val": sorted(p.name for p in (export_root / subdir / "val").iterdir()),
    }


def test_export_layout_and_correctness(tmp_path):
    work_dir = tmp_path / "work"
    export_dir = tmp_path / "dataset"
    manifest = _build_manifest_and_files(work_dir)

    summary = write_dataset(
        manifest,
        classes=CLASSES,
        work_dir=str(work_dir),
        export_dir=str(export_dir),
        formats=["yolo", "coco"],
        val_split=0.0,  # keep all in train so we can assert exact membership simply
        seed=42,
    )

    export_root = Path(export_dir)

    # --- §6.7 layout exists --------------------------------------------------
    assert (export_root / "data.yaml").is_file()
    for split in ("train", "val"):
        assert (export_root / "images" / split).is_dir()
        assert (export_root / "labels" / split).is_dir()
    assert (export_root / "annotations" / "instances_train.json").is_file()
    assert (export_root / "annotations" / "instances_val.json").is_file()

    # --- skipped image EXCLUDED everywhere -----------------------------------
    img_files = _all_split_files(export_root, "images")
    all_imgs = img_files["train"] + img_files["val"]
    assert "img_a.jpg" in all_imgs
    assert "img_b.jpg" in all_imgs
    assert "img_c.jpg" not in all_imgs, "skipped image must be excluded"
    # Two images included -> summary counts agree.
    assert summary["train"] + summary["val"] == 2
    assert summary["boxes"] == 2  # only img_a's two boxes (img_c excluded)

    # --- YOLO labels: values in [0,1], class ids in range --------------------
    label_root = export_root / "labels"
    # img_a has two boxes -> non-empty file.
    a_txt = next(p for p in label_root.rglob("img_a.txt"))
    a_lines = [ln for ln in a_txt.read_text().splitlines() if ln.strip()]
    assert len(a_lines) == 2
    for line in a_lines:
        parts = line.split()
        assert len(parts) == 5
        class_id = int(parts[0])
        assert 0 <= class_id < len(CLASSES)  # in range
        for val in map(float, parts[1:]):
            assert 0.0 <= val <= 1.0  # normalized YOLO values

    # img_b has ZERO boxes -> file exists and is EMPTY.
    b_txt = next(p for p in label_root.rglob("img_b.txt"))
    assert b_txt.read_text().strip() == "", "zero-box image must yield an empty label file"

    # No label file for the skipped image.
    assert not list(label_root.rglob("img_c.txt"))

    # --- COCO JSON valid, bbox absolute px within bounds ---------------------
    # img_a is in some split; find which COCO doc holds it.
    coco_docs = {
        "train": json.loads((export_root / "annotations" / "instances_train.json").read_text()),
        "val": json.loads((export_root / "annotations" / "instances_val.json").read_text()),
    }
    # Categories mirror the class list with matching ids in both docs.
    for doc in coco_docs.values():
        assert doc["categories"] == [{"id": 0, "name": "mandarin"}]

    # Locate img_a's COCO image entry + its annotations.
    img_a_entry = None
    holding_split = None
    for split, doc in coco_docs.items():
        for im in doc["images"]:
            if im["file_name"] == "img_a.jpg":
                img_a_entry = im
                holding_split = split
    assert img_a_entry is not None
    assert (img_a_entry["width"], img_a_entry["height"]) == (800, 600)

    anns = [
        a for a in coco_docs[holding_split]["annotations"]
        if a["image_id"] == img_a_entry["id"]
    ]
    assert len(anns) == 2
    W, H = 800, 600
    # Hand-check the right-half box: COCO bbox must be [400,60, 320,180] in PIXELS.
    #   x_min = 0.5*800 = 400 (NOT 0.5*600=300, the W/H-swap value)
    #   y_min = 0.1*600 = 60 ; width = 0.4*800 = 320 ; height = 0.3*600 = 180
    boxes = [tuple(a["bbox"]) for a in anns]
    # Float tolerance: 0.3*600 lands at 180.00000000000003, not exactly 180.0.
    assert any(
        b == pytest.approx((400.0, 60.0, 320.0, 180.0), abs=1e-6) for b in boxes
    ), f"expected right-half COCO bbox in pixels, got {boxes}"
    for a in anns:
        x, y, bw, bh = a["bbox"]
        assert a["category_id"] == 0
        # Absolute pixels within image bounds.
        assert 0.0 <= x and 0.0 <= y
        assert x + bw <= W + 1e-6
        assert y + bh <= H + 1e-6
        assert a["area"] == pytest.approx(bw * bh)

    # COCO must NOT reference the skipped image.
    for doc in coco_docs.values():
        assert all(im["file_name"] != "img_c.jpg" for im in doc["images"])

    # --- data.yaml class metadata -------------------------------------------
    import yaml

    data = yaml.safe_load((export_root / "data.yaml").read_text())
    assert data["nc"] == 1
    assert data["names"] == ["mandarin"]


def test_export_split_is_deterministic(tmp_path):
    """Same seed + same images => identical train/val membership across two runs."""
    work_dir = tmp_path / "work"

    # Build enough images that a nonzero val split actually places some in val,
    # so the determinism check is meaningful (not trivially {} == {}).
    images_dir = work_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for i in range(10):
        fname = f"im_{i}.jpg"
        Image.new("RGB", (640, 480), (i * 10, 30, 60)).save(images_dir / fname)
        records.append(
            ImageRecord(
                id=f"im_{i}",
                path=f"images/{fname}",
                width=640,
                height=480,
                status="pending",
                detections=[
                    Detection(label="mandarin", box=BBox(x1=0.2, y1=0.2, x2=0.5, y2=0.5), score=0.6)
                ],
            )
        )
    manifest = Manifest.new(
        records,
        detector_backend="grounding_dino",
        ontology={"mandarin orange on tree": "mandarin"},
        box_threshold=0.30,
        text_threshold=0.25,
        tiling=False,
        grove_version=grove.__version__,
    )

    def run(export_dir: Path) -> dict[str, list[str]]:
        write_dataset(
            manifest,
            classes=CLASSES,
            work_dir=str(work_dir),
            export_dir=str(export_dir),
            formats=["yolo", "coco"],
            val_split=0.3,  # 30% of 10 -> 3 images in val
            seed=123,
        )
        return _all_split_files(export_dir, "images")

    first = run(tmp_path / "dataset1")
    second = run(tmp_path / "dataset2")

    # A nonzero val split actually happened (guards against a trivial pass).
    assert len(first["val"]) == 3
    assert len(first["train"]) == 7
    # Deterministic: identical membership in both splits across runs.
    assert first == second
