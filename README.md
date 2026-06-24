# Grove — Mandarin Auto-Labeling Pipeline

Grove turns a folder of mandarin-tree photos into **object-detection training data**
(YOLO + COCO). An open-vocabulary detector draws draft boxes, a human corrects them in a
local browser UI, and the corrected labels are exported as a clean dataset — the factory
that produces training labels for a mandarin-picking robot's perception model.

> **The deliverable is label data (coordinate files), not annotated images.** The previews
> exist only so a human can verify the labels. See `CLAUDE.md` for the full build spec.

If you are an operator (not a developer), you want **`OPERATOR_GUIDE.md`**, not this file.

---

## Architecture at a glance

Grove is split along the **GPU boundary**. The two halves talk *only through files on disk*
(the working/dataset directory), which is what lets the heavy half run in a free cloud GPU
notebook and the light half run on a laptop with no GPU.

```
                       ┌──────────────── shared core (grove.core) ────────────────┐
                       │  models.py  canonical BBox / Detection / ImageRecord /     │
                       │             Manifest (Pydantic)                            │
                       │  config.py  one validated YAML -> Config                   │
                       │  formats.py canonical <-> YOLO / COCO / pixel converters   │
                       │             (GPU-free, dependency-light, unit-tested)      │
                       └───────────────────────────────────────────────────────────┘
        GPU side (Kaggle/Colab) — grove.pipeline          Local side (Mac) — grove.review
        ────────────────────────────────────────          ──────────────────────────────────
        ingest    folder        -> manifest.json           server.py  FastAPI endpoints
        detect    manifest      -> predictions.json         store.py   corrections (authoritative)
        annotate  predictions   -> preview images           static/    canvas UI (create/move/
        export    state         -> draft YOLO + COCO                   resize/delete/relabel)
        train     reviewed set  -> student YOLO (optional)   -> re-export final YOLO + COCO
```

**Canonical coordinate format** (the only thing the code reasons in): normalized `xyxy`,
top-left origin, every value in `[0, 1]`. Conversions happen *only at the boundaries*
(`grove/core/formats.py`). This is the easiest place to introduce silent bugs, so the
converters are unit-tested first (`tests/test_formats.py`).

**The detector is a teacher, not the robot.** Grounding DINO (via `autodistill`) is accurate
but slow — it *labels* data. The optional `train` stage distills those reviewed labels into a
fast YOLO **student**, which is the only robot-deployable artifact.

---

## End-to-end data flow

```
raw images
  -> data/work/manifest.json          (ingest:   scan + EXIF-normalize + stable ids)
  -> data/work/predictions.json       (detect:   open-vocab detector, optional tiling)
  -> data/work/previews/*             (annotate: boxes drawn for human QC)
  -> data/dataset/ (draft YOLO+COCO)  (export)
  --- download the dataset, review locally ---
  -> data/work/review_store.json      (review:   human corrections, kept separate from predictions)
  -> data/dataset/ (final YOLO+COCO)  (review re-export — the real deliverable)
  -> data/work/runs/ (student YOLO)   (train, optional)
```

Prepared, EXIF-normalized images live at `data/work/images/<id>.<ext>`; every
`ImageRecord.path` is stored **relative to `work_dir`** so the dataset is portable between the
cloud and the laptop.

---

## Repository layout

```
grove/
├── core/                  # GPU-free shared core (safe to import anywhere)
│   ├── models.py          # canonical BBox / Detection / ImageRecord / Manifest
│   ├── config.py          # Pydantic config schema + load_config()
│   └── formats.py         # canonical <-> YOLO / COCO / pixel converters
├── pipeline/              # GPU side
│   ├── ingest.py          # folder -> Manifest (EXIF normalize, stable ids)
│   ├── detectors/
│   │   ├── base.py        # Detector protocol + get_detector() registry
│   │   ├── grounding_dino.py
│   │   └── yolo_world.py
│   ├── tiling.py          # slice -> detect per tile -> merge (NMS)  [SAHI strategy]
│   ├── detect.py          # run a detector over the manifest (resumable)
│   ├── annotate.py        # draw boxes for QC (supervision)
│   ├── export.py          # write YOLO + COCO (deterministic split, validated)
│   └── train.py           # optional: distill student YOLO (ultralytics)
├── review/                # local side (no GPU)
│   ├── server.py          # FastAPI app + endpoints
│   ├── store.py           # annotation store (source of truth for corrections)
│   └── static/            # index.html, app.js, styles.css (canvas UI)
└── cli.py                 # grove ingest|detect|annotate|export|train|review

notebooks/kaggle_label.ipynb   # single Run-All GPU job (install -> ... -> zip dataset)
tests/                          # test_formats (first), test_tiling, test_ingest, test_export
config.example.yaml             # documented config template (copy to config.yaml)
setup.sh / Makefile             # one-command local setup + convenience targets
```

---

## CLI

All subcommands read the same YAML (`--config config.yaml`, default `config.yaml`):

```
grove ingest    # folder -> manifest.json
grove detect    # manifest -> predictions.json            (GPU; resumable)
grove annotate  # predictions -> preview images           (GPU)
grove export    # current state -> YOLO + COCO
grove train     # reviewed dataset -> student YOLO         (GPU; optional)
grove review    # launch the local FastAPI review UI       (no GPU)
```

`grove review` imports only the light `[review]` stack, so it runs on a Mac with no GPU.

---

## Configuration

One YAML drives everything (see `config.example.yaml`). The operator normally edits **only the
paths and the detector ontology**; every other field has a working default.

```yaml
detector:
  backend: grounding_dino          # grounding_dino | grounding_dino_hf | yolo_world  (swappable, §2.5)
  ontology:
    "tree trunk": tree             # PROMPT TEXT -> CLASS NAME (ships targeting individual trees)
  box_threshold: 0.20
  text_threshold: 0.15
tiling:
  enabled: false                   # turn on for small/distant fruit in wide shots
export:
  formats: [yolo, coco]
  val_split: 0.15
  seed: 42                         # deterministic split
```

Prompt phrasing and `box_threshold` are dataset-sensitive — see `CLAUDE.md` §8/§11 for tuning.

---

## Developing

```bash
./setup.sh          # venv + GPU-free [review,dev] install
make test           # run the GPU-free test suite (converters, tiling, ingest, export)
```

**Continuous integration.** `.github/workflows/ci.yml` runs the full test suite on every push
and PR to `main`, across Python 3.10/3.11/3.12. It installs the light `[review,dev]` stack plus
`supervision`, so even the detector-adapter tests run on a plain CPU runner — the GPU backends'
`supervision -> canonical` conversion, the registry/construction dispatch, and the autodistill ->
HF fallback are all exercised with autodistill mocked in-process. Only real model forward passes
(which need CUDA + weights) skip themselves.

- The **core** stays dependency-light (pydantic + pyyaml only) so it imports on any machine.
- GPU-heavy backends (`autodistill`, `supervision`, `ultralytics`) live behind the `[gpu]`
  extra and are imported lazily, so importing the CLI or the review server never pulls in torch.
- The detector backend is **stubbed** in tests, so the pipeline and exporters are testable
  without a GPU.

Pinned dependency groups (`pyproject.toml`): base (light) · `[review]` (FastAPI, torch-free) ·
`[gpu]` (Kaggle) · `[dev]` (pytest, httpx).

See `CLAUDE.md` for the authoritative spec, build milestones (M0–M5), and design rationale.
