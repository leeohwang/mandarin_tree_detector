# SPEC.md — Grove: Mandarin Auto-Labeling Pipeline

> This file is the single source of truth for building this project. You are
> starting with **no prior context**. Read this document fully before writing any code.
> When the spec is silent on a decision, follow the *principles* stated here rather than
> guessing, and prefer the simplest thing that satisfies them.
>
> **Two standing requirements that shape every decision:** (1) Keep the human operator's
> manual steps to an absolute minimum — automate everything that *can* be automated (install,
> weight download, format conversion, dataset packaging) and leave the human only the few
> actions that genuinely cannot be scripted. (2) Produce and maintain **`OPERATOR_GUIDE.md`**,
> a short, ordered checklist of exactly those irreducible manual actions. Fewer steps in that
> file is a primary success metric. Full spec in §10.4.

---

## 1. What you are building and why

**Mission:** Build a tool that takes a folder of photographs of mandarin (citrus) trees,
automatically draws bounding boxes around every mandarin fruit, lets a human review and
correct those boxes, and exports the result as **object-detection training data** (YOLO and
COCO format).

**The real downstream goal:** This training data feeds the perception model of an automated
agricultural **mandarin-picking robot**. The robot needs a fast, accurate detector that can
find fruit on a tree in real time. We are not building that detector's runtime here — we are
building the *factory that produces its training labels*.

**The single most important reframe — internalize this:**

> The deliverable of this project is **label data (coordinate files)**, not pretty annotated
> images. The annotated images exist only so a human can verify the labels are good. If you
> ever find yourself optimizing the visual output over the correctness and format of the
> exported coordinates, you have lost the plot.

**Why an open-vocabulary detector and not an LLM:** General multimodal LLMs do not render
boxes; they emit coordinates as text, inconsistently, and they localize small/occluded fruit
poorly. They also cost money per image and impose rate limits. An **open-vocabulary object
detector** (you prompt it with the word "mandarin" and it returns boxes) is open-source, runs
with no per-image cost and no rate limits, and produces far tighter, more consistent boxes.
This choice is final. Do not add an LLM detection backend.

---

## 2. Critical context — non-obvious facts that shape the whole design

Read all five. Each one has caused real mistakes when ignored.

1. **The detector is a "teacher," not the robot's model.** We use **Grounding DINO** (via the
   `autodistill` library) to *label* data. Grounding DINO is accurate but slow — unsuitable
   for a robot's real-time camera. The pipeline optionally *distills* its labels into a fast
   **YOLO** "student" model, which is what would eventually run on the robot. Never treat
   Grounding DINO as a deployable runtime detector.

2. **Human review is mandatory, not optional.** Auto-labels on a hard crop like clustered,
   occluded, backlit fruit *will* contain misses and false positives. Training a final model
   on raw, unreviewed auto-labels produces a robot that misses fruit or grabs leaves. The
   auto-label pass produces **drafts**; a human correction step turns drafts into a dataset.
   Build the review UI as a first-class component, not an afterthought.

3. **The compute constraint is real.** The target user develops on an **Intel Mac with no
   CUDA GPU**. Grounding DINO needs a GPU to be usable at folder scale (on CPU it is tens of
   seconds per image). Therefore the architecture **splits along the GPU boundary**: the
   GPU-bound labeling/training runs in a cloud notebook (Kaggle, free tier — ~30 GPU
   hrs/week, P100/T4); the GPU-free review UI runs locally on the Mac. Design every component
   to respect which side of that boundary it lives on. The `grove` package must be
   pip-installable so the same code runs in Kaggle and locally.

4. **Coordinate conventions are a minefield.** Every format expresses boxes differently
   (YOLO = normalized center+size; COCO = absolute pixel corner+size; Grounding
   DINO/`supervision` = absolute pixel corners). Bugs here are silent — boxes come out
   mirrored, transposed, or off by a constant. **Adopt ONE canonical internal format and
   convert only at the boundaries.** See §9. This is the easiest place to lose a day; treat
   the converters as critical code and unit-test them first.

5. **Honor the "swappable backend" requirement.** The original product vision called for
   swapping detection models. With open-vocab detectors there are no API keys to rotate, but
   the *spirit* — pluggability — stays. Build a `Detector` abstraction so the backend
   (Grounding DINO, YOLO-World, future models) is chosen by config, and the rest of the code
   never knows which one ran.

---

## 3. System architecture

Two subsystems sharing a core. The dividing line is the GPU boundary.

```
                         ┌─────────────────────────── shared core ───────────────────────────┐
                         │  grove.core: data model (canonical boxes), config, format          │
                         │  converters (canonical ↔ YOLO ↔ COCO). No GPU, no heavy deps.       │
                         └────────────────────────────────────────────────────────────────────┘
                                 ▲                                              ▲
                                 │                                              │
   ── GPU SIDE (Kaggle/Colab) ───┴──────────────────         ── LOCAL SIDE (Mac, no GPU) ──┴───────────────
                                                                                                      
   grove.pipeline (CLI):                                       grove.review (FastAPI + canvas UI):       
     ingest  → build image manifest                              loads a labeled dataset                  
     detect  → open-vocab detector → predictions                 shows image + overlaid boxes on a canvas 
               (optional tiling for small/dense fruit)           human adds / moves / deletes / relabels  
     annotate→ draw boxes on images for visual QC                marks each image reviewed / skipped      
     export  → write YOLO + COCO draft datasets                  saves corrections; re-exports YOLO/COCO  
     train   → (optional) distill student YOLO                                                            
```

**End-to-end data flow:**

```
raw image folder
   → manifest.json                (ingest)
   → predictions (canonical)      (detect)
   → annotated previews + draft YOLO/COCO   (annotate, export)
   → [transfer dataset to local machine]
   → human-corrected annotations  (review UI)
   → final YOLO/COCO              (review UI re-export)
   → student YOLO weights         (train, optional)
```

The two subsystems communicate **only through files on disk** (the dataset directory). They
never call each other directly. This is what lets the heavy half run in Kaggle and the light
half run on the Mac.

---

## 4. Tech stack

Pin major libraries in `pyproject.toml`. Rationale matters — do not silently substitute.

| Concern | Choice | Why |
|---|---|---|
| Language | Python 3.10+ | `autodistill`/`ultralytics` baseline; modern typing |
| Labeling engine | `autodistill` + `autodistill-grounding-dino` | Wraps Grounding DINO; `.label()` does folder-scale auto-labeling in a few lines |
| Alt backend | `autodistill-yolo-world` | Second open-vocab detector to prove the abstraction |
| Box drawing & dataset utils | `supervision` | Renders boxes; `Detections` type; dataset converters |
| Student training | `ultralytics` | Trains the YOLO student (v8/v11/26); same exported format |
| Image I/O | `opencv-python`, `Pillow` | Read/write/resize; EXIF handling |
| Tiling (small fruit) | implement with `supervision` slicing, or `sahi` | Slice large tree images so distant fruit aren't lost |
| Config | `pydantic` + `PyYAML` | Typed, validated config loaded from a single YAML |
| Data models | `pydantic` | Canonical box / detection / manifest schemas |
| Review backend | `fastapi` + `uvicorn` | Lightweight local server; no GPU |
| Review frontend | **vanilla HTML/CSS/JS + `<canvas>`** | Box editing IS canvas work; zero build tooling. See §6.6 |
| CLI | `typer` (or `argparse`) | Subcommands map to pipeline stages |
| Tests | `pytest` | Converters and tiling especially |

When you build the review UI's frontend, **consult the `frontend-design` skill** for styling
conventions before writing markup. Keep it clean and legible; this is a tool for staring at
hundreds of images, so prioritize clarity and fast keyboard interaction over decoration.

---

## 5. Repository structure

Create this layout. Keep `core` dependency-light so it imports cleanly on the Mac without GPU
libraries.

```
grove/
├── SPEC.md                      # this file (build spec)
├── README.md                      # dev overview: architecture + how the code is organized
├── OPERATOR_GUIDE.md              # ★ the human's manual run steps — keep MINIMAL (see §10.4)
├── setup.sh                       # one-command local setup (venv + [review] install)
├── pyproject.toml                 # package + pinned deps; extras: [gpu], [review]
├── config.example.yaml            # documented config template
│
├── grove/
│   ├── __init__.py
│   ├── core/                      # NO GPU, NO heavy deps — safe to import anywhere
│   │   ├── models.py              # Pydantic: BBox, Detection, ImageRecord, Manifest
│   │   ├── config.py              # Pydantic config schema + YAML loader
│   │   └── formats.py             # canonical ↔ YOLO ↔ COCO converters (critical, tested)
│   │
│   ├── pipeline/                  # GPU side
│   │   ├── ingest.py              # scan folder → Manifest
│   │   ├── detectors/
│   │   │   ├── base.py            # Detector Protocol + registry
│   │   │   ├── grounding_dino.py  # default backend (autodistill)
│   │   │   └── yolo_world.py      # alt backend
│   │   ├── tiling.py              # slice → detect per tile → merge (NMS)
│   │   ├── detect.py              # orchestrates a detector over the manifest
│   │   ├── annotate.py            # draw boxes onto images for QC (supervision)
│   │   ├── export.py              # write YOLO + COCO datasets
│   │   └── train.py               # optional: distill student YOLO (ultralytics)
│   │
│   ├── review/                    # LOCAL side
│   │   ├── server.py              # FastAPI app + endpoints
│   │   ├── store.py               # annotation store (source of truth for corrections)
│   │   └── static/                # index.html, app.js, styles.css (canvas UI)
│   │
│   └── cli.py                     # `grove ingest|detect|annotate|export|train|review`
│
├── notebooks/
│   └── kaggle_label.ipynb         # thin wrapper: install grove, run detect+annotate+export
│
└── tests/
    ├── test_formats.py            # round-trip canonical↔YOLO↔COCO (write these FIRST)
    ├── test_tiling.py             # tile boundaries, merge/NMS correctness
    └── test_ingest.py
```

---

## 6. Component specifications

### 6.1 Core data model (`grove/core/models.py`)

Define with Pydantic:

- **`BBox`** — the canonical box. Fields: `x1, y1, x2, y2` as floats in `[0, 1]`, **top-left
  origin**, normalized to image size. Invariants: `0 ≤ x1 < x2 ≤ 1`, `0 ≤ y1 < y2 ≤ 1`.
  Provide helpers: `.width`, `.height`, `.area`, `.clip()` (clamp to [0,1]).
- **`Detection`** — `label: str`, `box: BBox`, `score: float | None`.
- **`ImageRecord`** — `id: str` (stable, derived from relative path), `path: str`,
  `width: int`, `height: int` (pixels), `detections: list[Detection]`, `status: Literal["pending","reviewed","skipped"]`.
- **`Manifest`** — `images: list[ImageRecord]`, plus run metadata: detector name, prompt/
  ontology, thresholds, model version, timestamp. **Every run must record its provenance** so
  any dataset is traceable back to the exact settings that produced it.

### 6.2 Configuration (`grove/core/config.py`, `config.example.yaml`)

One YAML drives everything. Schema (validate with Pydantic):

```yaml
project_name: mandarin-orchard-v1

paths:
  input_dir: ./data/raw            # folder of tree images
  work_dir:  ./data/work           # manifest, predictions, previews
  export_dir: ./data/dataset       # final YOLO/COCO output

detector:
  backend: grounding_dino          # grounding_dino | yolo_world
  # ontology maps PROMPT TEXT -> CLASS NAME written into labels
  ontology:
    "mandarin orange on tree": mandarin
  box_threshold: 0.30              # detection confidence; tune (see §8)
  text_threshold: 0.25

tiling:
  enabled: false                   # turn on for small/dense fruit in large images
  tile_size: 640
  overlap: 0.20
  nms_iou: 0.50                    # merge duplicate detections across tile seams

classes:                           # canonical class list and IDs (YOLO needs stable ids)
  - mandarin

export:
  formats: [yolo, coco]
  val_split: 0.15                  # train/val split for the exported dataset

train:                             # optional student-distillation stage
  enabled: false
  model: yolov8n.pt
  epochs: 100
  imgsz: 640
```

### 6.3 Ingest (`pipeline/ingest.py`)

Scan `input_dir` recursively for images (`.jpg/.jpeg/.png/.webp`). For each: read pixel
dimensions, normalize EXIF orientation (rotate pixels so stored orientation is upright, then
drop the EXIF tag — otherwise boxes and image will disagree), assign a stable `id` from the
relative path. Emit `manifest.json`. Skip and log unreadable/corrupt files rather than
crashing the run.

### 6.4 Detector abstraction (`pipeline/detectors/`)

```python
# base.py
class Detector(Protocol):
    def detect(self, image_bgr: "np.ndarray") -> list[Detection]: ...
    # returns Detections with boxes already in CANONICAL format (normalized xyxy)

# A simple registry maps config `backend` -> Detector class.
def get_detector(cfg) -> Detector: ...
```

Each backend's job is identical from the outside: take an image, return canonical
`Detection`s. Internally it adapts the library's native output.

- **`grounding_dino.py`** (default): construct an `autodistill` `GroundingDINO` from the
  ontology and thresholds. Its output is a `supervision` `Detections` object with **`xyxy` in
  absolute pixels**; convert to canonical by dividing x by image width and y by height. Map
  class ids back to names via the ontology.
- **`yolo_world.py`**: same contract via `autodistill-yolo-world`. Exists primarily to prove
  the abstraction holds; keep it thin.

> Do not let backend-specific coordinate quirks leak past this layer. Everything downstream
> sees only canonical `Detection`s.

### 6.5 Tiling (`pipeline/tiling.py`)

For large tree images, distant fruit can be only a few pixels wide and get missed at full
downscale. When `tiling.enabled`, slice the image into overlapping tiles (`tile_size`,
`overlap`), run the detector on each tile, translate each tile's boxes back into full-image
coordinates, and merge across seams with NMS (`nms_iou`). This is the SAHI strategy. Keep it a
clean wrapper *around* any `Detector` so it works regardless of backend. Unit-test the
coordinate translation and the seam-merge: an object spanning a tile boundary must end up as
one box, not two.

### 6.6 Annotate (`pipeline/annotate.py`)

Use `supervision`'s box annotator to draw canonical boxes (converted to pixels) onto copies of
the images, writing previews to `work_dir/previews/`. Include the label and score on each box.
**These are for human QC only** — they are never the training output. Keep this stage cheap and
optional.

### 6.7 Export (`pipeline/export.py`)

Write the dataset in the configured formats from the **current annotation state** (predictions
pre-review, or corrections post-review — the exporter doesn't care which, it reads the store/
manifest). Produce a standard layout:

```
export_dir/
├── data.yaml            # YOLO: class names, paths, nc
├── images/{train,val}/
├── labels/{train,val}/  # YOLO .txt per image
└── annotations/
    ├── instances_train.json   # COCO
    └── instances_val.json
```

Apply `val_split` deterministically (seed it) so runs are reproducible. Validate on the way
out: every label references a real class id; no box coordinates fall outside the image.

### 6.8 Review UI (`grove/review/`) — the human correction loop

This is the component that turns drafts into a dataset. Runs locally; no GPU.

**Annotation store (`store.py`):** the source of truth for human corrections, seeded from the
detector predictions on first load. For the MVP a single JSON document (or SQLite if you
prefer durability) keyed by image id, holding current boxes + review status. Keep it **separate
from the raw predictions file** so corrections never destroy the original auto-labels — useful
for measuring how much the model got wrong.

**Backend (`server.py`, FastAPI):**

| Method & path | Purpose |
|---|---|
| `GET /api/images` | list image ids + status + box counts (for the queue/progress) |
| `GET /api/images/{id}` | image metadata + current boxes (canonical) |
| `GET /api/images/{id}/file` | raw image bytes |
| `PUT /api/images/{id}/boxes` | replace this image's boxes with the corrected set |
| `POST /api/images/{id}/status` | mark `reviewed` / `skipped` |
| `POST /api/export` | re-export YOLO/COCO from corrected state |

**Frontend (`static/`, vanilla JS + canvas):**
- Render the current image to a `<canvas>`; draw existing boxes as an overlay.
- **Create** a box by click-drag; **select** by click; **move/resize** via corner handles;
  **delete** with Backspace/Delete; **relabel** via a dropdown (from the class list).
- **Navigate** prev/next with arrow keys; a sidebar shows the image queue with per-image
  status and overall progress (e.g. "47 / 320 reviewed").
- Keyboard-first: reviewing hundreds of images by mouse alone is painful. At minimum:
  next/prev, mark-reviewed, new-box mode, delete.
- Persist on every change (or on navigation) via the API. Convert canvas pixel coords ↔
  canonical at the UI boundary only.

> If building this UI is more than the user wants to take on, note the escape hatch in §11 —
> but the spec's default is to build it, because it gives full control and is the heart of
> producing trustworthy labels.

### 6.9 Train (`pipeline/train.py`) — optional student distillation

When `train.enabled`, train an `ultralytics` YOLO on the **reviewed** exported dataset. This is
the teacher→student distillation: Grounding DINO's reviewed labels become a fast YOLO suited to
the robot. Save weights and validation metrics under `work_dir/runs/`. This stage is GPU-side
(Kaggle). Make clear in logs that the resulting weights — not Grounding DINO — are the
robot-deployable artifact.

---

## 7. CLI (`grove/cli.py`)

Subcommands map one-to-one to pipeline stages; all read the same config:

```
grove ingest    --config config.yaml      # folder → manifest
grove detect    --config config.yaml      # manifest → predictions  (GPU)
grove annotate  --config config.yaml      # predictions → preview images
grove export    --config config.yaml      # current state → YOLO + COCO
grove train     --config config.yaml      # reviewed dataset → student YOLO  (GPU, optional)
grove review    --config config.yaml      # launch local FastAPI review UI
```

`grove detect` should be safe to re-run and resumable where practical (skip images already
predicted) so a dropped Kaggle session doesn't waste the whole quota.

---

## 8. Coordinate systems and format conversions (§9 — read carefully)

**Canonical (internal, the only thing the codebase reasons in):**
normalized **xyxy**, top-left origin. `box = (x1, y1, x2, y2)`, each in `[0, 1]`.

Conversions, given image pixel size `W × H`:

**Canonical → YOLO** (`class_id cx cy w h`, all normalized, center-based):
```
cx = (x1 + x2) / 2
cy = (y1 + y2) / 2
w  =  x2 - x1
h  =  y2 - y1
```

**YOLO → Canonical:**
```
x1 = cx - w/2 ;  x2 = cx + w/2
y1 = cy - h/2 ;  y2 = cy + h/2
```

**Canonical → COCO** (`bbox = [x_min, y_min, width, height]`, **absolute pixels**):
```
x_min = x1 * W ;  width  = (x2 - x1) * W
y_min = y1 * H ;  height = (y2 - y1) * H
```

**COCO → Canonical:**
```
x1 = x_min / W ;  x2 = (x_min + width)  / W
y1 = y_min / H ;  y2 = (y_min + height) / H
```

**Grounding DINO / `supervision` → Canonical:** native `xyxy` is **absolute pixels**; divide
x-values by `W`, y-values by `H`.

**Canvas (review UI) ↔ Canonical:** multiply/divide by the rendered canvas dimensions, and
account for any display scaling between natural image size and canvas size.

> Write `tests/test_formats.py` **first**, before the converters are used anywhere: assert that
> canonical → YOLO → canonical and canonical → COCO → canonical both round-trip to within
> floating-point tolerance, on non-square images (so any W/H swap is caught immediately).

---

## 9. Build order (milestones)

Build incrementally; each milestone is independently runnable and verifiable. Do not skip
ahead — later stages depend on the data contracts established earlier.

- **M0 — Scaffolding & core.** Package skeleton, `pyproject.toml` with extras, Pydantic
  `models.py` and `config.py`, `formats.py` converters. **Write and pass `test_formats.py`.**
  Nothing GPU-bound yet; this all runs on the Mac.
- **M1 — Auto-label loop.** `ingest` + the Grounding DINO `Detector` + `detect` + `annotate`.
  Deliver `notebooks/kaggle_label.ipynb` that installs `grove`, runs these on a sample folder,
  and shows preview images inline. **This is the first "it works" moment** — boxes on
  mandarins, produced on a free Kaggle GPU. Also write the first `OPERATOR_GUIDE.md` capturing
  the (minimal) manual steps needed to reach this point.
- **M2 — Export.** YOLO + COCO writers with deterministic split and on-write validation.
- **M3 — Review UI.** Store, FastAPI endpoints, canvas frontend with create/move/resize/
  delete/relabel + keyboard nav + progress + re-export. The correction loop closes here.
- **M4 — Student training (optional).** `train.py` distilling a YOLO on the reviewed dataset.
- **M5 — Domain tuning & polish.** Tiling backend, threshold sweeps, the second (`yolo_world`)
  detector to validate pluggability, README, resumable `detect`, and a finalized
  `OPERATOR_GUIDE.md` trimmed to the fewest possible manual steps.

A good first PR is **M0 + M1**: it proves the GPU path and the core data contracts together.

---

## 10. Environment & running

- **GPU side (pipeline):** primary target is a **Kaggle notebook** — free, ~30 GPU hrs/week
  on P100/T4, 9-hour sessions, supports background execution. Colab free tier works as backup
  but disconnects more readily. The notebook should `pip install -e .` (or install from the
  repo) and call the CLI/stages. Because `detect` is the GPU-hungry stage, make it resumable
  so a dropped session doesn't burn the weekly quota.
- **Local side (review UI):** runs on the Intel Mac with no GPU. `grove review` starts FastAPI
  on `localhost`; the user reviews in the browser. The `[review]` install extra must not pull
  in GPU/torch-heavy dependencies, so the Mac install stays light.
- **Dataset hand-off:** the GPU side writes a dataset directory; the user downloads it from
  Kaggle; the review UI opens it locally. Files on disk are the only interface between halves.

### 10.4 Minimal manual setup + the `OPERATOR_GUIDE.md` deliverable

**Engineering mandate — drive the human's manual steps toward zero.** Before documenting any
manual step, ask whether it can be automated, and if so, automate it instead of writing it
down. Concretely:

- Bundle the entire GPU run into the **one** `notebooks/kaggle_label.ipynb` so "run it" is a
  single **Run All**: install, weight download, ingest, detect, annotate, export, and
  **auto-zip the dataset** for one-click download all happen inside that notebook.
- Provide a **one-command local setup** — `setup.sh` (and/or a `Makefile` with `make setup`,
  `make review`) that creates the venv and installs the `[review]` extra. Never hand the user
  a multi-line copy-paste sequence.
- Ship **working config defaults** so the user edits only the input/output paths, not a wall
  of options. Auto-create output directories rather than asking the user to.
- Add a **preflight check** at the top of the notebook that verifies GPU + internet are on and
  prints a precise one-line fix if not. It cannot toggle them itself, but it must remove all
  guesswork.
- Model weights download automatically on first use — never ask the user to fetch weights.

**Required deliverable: `OPERATOR_GUIDE.md`.** A standalone, human-facing file containing
*only* the irreducible manual actions — the things that genuinely cannot be scripted. It is
not architecture and not dev docs (those live in `README.md`); it is a checklist the user
follows to go from zero to a finished dataset. Requirements:

- **Ordered and copy-pasteable.** Exact commands in fenced code blocks; exact UI actions
  ("click X → Y") for anything that is not a command.
- **Three short sections only:** (a) **One-time setup** (done once, ever), (b) **Each labeling
  run — Kaggle / GPU side**, (c) **Each review run — local / Mac side**.
- **As short as the automation allows.** Every line is a step the user must personally perform.
  If a step can be removed by automating it, remove it — treat the length of this file as a bug
  count to drive down.
- **No internal detail.** The user does not need to know how converters or tiling work; only
  what to do.

As a baseline, the irreducible manual steps this project currently has are roughly the list
below. Automate anything beyond these, and keep the guide to this shape:

```
One-time setup
  1. Create a Kaggle account and PHONE-VERIFY it (unlocks GPU + internet; cannot be scripted).
  2. Clone the repo and run the one-command local setup (./setup.sh  or  make setup).

Each labeling run (Kaggle / GPU)
  1. Upload the image folder to the notebook (or as a Kaggle Dataset).
  2. Set Accelerator → GPU and Internet → On in the notebook sidebar.
  3. Run All, then download the produced dataset .zip.

Each review run (local / Mac)
  1. Put the downloaded dataset where config points (or pass --config).
  2. make review  → correct boxes in the browser → click Export.
```

Create a first `OPERATOR_GUIDE.md` at **M1** (when the pipeline first runs end-to-end on
Kaggle) and finalize it at **M5**. If a feature you add *introduces* a new manual step, prefer
redesigning so it doesn't.

---

## 11. Domain-specific guidance: detecting mandarins on trees

The crop makes this hard. Build with these realities in mind; expect to tune, not to nail it
first try.

- **Prompt sensitivity.** Open-vocab detectors are sensitive to the ontology phrasing. Don't
  assume one prompt is best — make it trivial to try alternatives via config: e.g. `"mandarin
  orange on tree"`, `"orange fruit"`, `"citrus fruit"`, `"round orange fruit on branch"`.
  Recall and false-positive rate shift noticeably between them.
- **Thresholds.** Start at `box_threshold ≈ 0.30`. Lowering it raises recall on small/partly
  hidden fruit but invites false positives on sunlit leaves, orange-ish background, and
  lens flare. The right value is dataset-specific; surface it in config and make sweeping easy.
- **Occlusion & clustering** are the core difficulty: fruit behind leaves, fruit touching
  fruit (boxes merge or split wrongly), fruit half out of frame. The detector will get these
  partly wrong — **this is precisely why the review step is mandatory.** Don't try to engineer
  the auto-labeler to perfection; engineer the *correction loop* to be fast.
- **Small/distant fruit** in wide tree shots: enable **tiling** (§6.5). It meaningfully
  recovers fruit that vanish at full-image downscale.
- **Lighting & background:** outdoor backlighting, bright sky behind the canopy, and dappled
  shade all stress the detector. Nothing to "fix" in code — just expect lower precision on
  those frames and lean on review.
- **Classes:** start with a **single class, `mandarin`.** Ripeness ("ripe"/"unripe") is
  tempting for the robot but is genuinely hard for open-vocab detection and adds reviewer
  burden — treat it as a future extension, not v1.

---

## 12. Pitfalls & gotchas (things that will bite)

- **Coordinate bugs are silent.** Mirrored/transposed/off-by-W boxes look "almost right."
  Trust the round-trip tests and eyeball previews on a *non-square* image early.
- **EXIF orientation.** If you don't normalize it at ingest, the image the model sees, the
  image the reviewer sees, and the stored boxes can disagree. Rotate pixels, drop the tag.
- **GPU OOM on free tiers.** Very large images can exhaust 16 GB VRAM. Resize or tile before
  detection; cap the working resolution in config if needed.
- **Treating auto-labels as final.** Never wire `detect → export → train` without the review
  step in between for a real dataset. The student model is only as good as the reviewed labels.
- **Deploying the teacher.** Grounding DINO is not the robot's runtime model. Only the trained
  YOLO student is deployable. Keep this distinction loud in code comments and logs.
- **Non-determinism.** Seed the train/val split and log detector settings per run, or two
  "identical" runs will produce subtly different datasets and you won't know why.
- **Frontend storage.** The review UI persists through the FastAPI backend (files/SQLite), not
  browser `localStorage`. Keep server-side state authoritative.

---

## 13. Testing & validation

- **Unit:** format round-trips (first priority), tiling coordinate translation + seam merge,
  ingest dimension/EXIF handling, config validation.
- **Integration:** a tiny fixture folder of a few images runs `ingest → detect (stub or real)
  → export` and produces a structurally valid YOLO + COCO dataset (right files, class ids in
  range, no out-of-bounds boxes).
- **Visual sanity:** an annotated-preview check on a non-square sample image — the cheapest way
  to catch coordinate bugs.
- The detector backend can be **stubbed** in tests (return fixed canonical boxes) so the
  pipeline and exporters are testable on the Mac without a GPU.

---

## 14. Out of scope for v1 (do not build now)

- The robot's real-time inference runtime / on-device deployment.
- Multi-class ripeness or species classification.
- Active-learning / model-in-the-loop retraining cycles.
- Multi-user review, authentication, or cloud-hosting the review UI.
- Cloud orchestration beyond a Kaggle/Colab notebook.

Keep the system a clean, single-user, file-based pipeline. Resist scope creep.

---

## 15. Alternatives & escape hatches

- **Don't want to build the review UI?** Mature tools — **Label Studio** and **CVAT** — import
  pre-labeled YOLO/COCO and provide a polished correction UI (and can even call a model for
  pre-labeling). The custom UI here exists for full control and as a learning exercise; if the
  user prefers, the pipeline can export drafts straight into Label Studio for review and skip
  `grove/review/`. The rest of the architecture is unchanged.
- **Need higher label quality / tighter masks?** A **Grounded-SAM** pipeline (Grounding DINO
  boxes → SAM segmentation) yields tight masks and instance separation. Treat as a future
  detector backend, not part of v1.

---

## 16. Definition of done (v1)

1. On a Kaggle GPU, `grove` auto-labels a folder of mandarin-tree photos and writes annotated
   previews plus a draft YOLO + COCO dataset.
2. Locally on the Mac, `grove review` opens that dataset, and a human can correct every box and
   re-export a clean YOLO + COCO dataset.
3. Format converters are unit-tested and round-trip on non-square images.
4. The detector backend is swappable via config (Grounding DINO default, YOLO-World proven).
5. (Optional) `grove train` distills a YOLO student on the reviewed dataset and reports
   validation metrics.
6. `OPERATOR_GUIDE.md` exists and lists *only* the irreducible manual steps (one-time setup +
   per-run Kaggle + per-run local), with everything automatable already automated (one-command
   local setup, single Run-All notebook, auto-zipped dataset, auto-downloading weights).

Build it in that order. Ship M0+M1 first.
