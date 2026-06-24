# Grove — Operator Guide

Three things to do. Nothing else is manual.

## 1. One-time setup
1. Create a Kaggle account and PHONE-VERIFY it (Settings -> Phone Verification). This unlocks the free GPU and Internet; it cannot be scripted.
2. Clone this repo and run the one-command local setup:
   ```bash
   ./setup.sh
   ```
3. Confirm your clone is wired correctly **before** spending any GPU quota (runs in seconds, no GPU needed):
   ```bash
   make selftest
   ```
   It runs the pipeline end-to-end (ingest -> detect -> export) with a stub detector. If it passes, the GPU run will work too.

> **What gets detected:** the shipped default detects **individual trees** (`"tree trunk" -> tree`). The Kaggle notebook reads `config.example.yaml` directly, and `setup.sh` copies it to a local `config.yaml` for review. To retarget (e.g. mandarin fruit), edit `detector.ontology` and `classes` in `config.example.yaml`, then copy it over your local config (`cp config.example.yaml config.yaml`) so the review side matches.

> **Make future runs zero-edit (do once, after you push to GitHub):** open
> `notebooks/kaggle_label.ipynb`, find `GROVE_REPO_URL = ""` near the top of the install cell, and
> paste your repo URL (e.g. `https://github.com/you/mandarin_tree_detector.git`). Now anyone can Run All
> with nothing to edit — the notebook clones the repo itself.

## 2. Each labeling run (Kaggle / GPU)
1. In the notebook's **Add Data** panel, attach your image folder as a Dataset. (The Grove repo is
   found automatically — via `GROVE_REPO_URL` if you set it above, or attach it as a Dataset too.)
   Image and repo paths auto-detect; you only set `INPUT_DIR` by hand if the guess is wrong.
2. Open `notebooks/kaggle_label.ipynb`. In the right sidebar set **Accelerator -> GPU** and **Internet -> On**.
3. Click **Run All**. When it finishes, download **`grove_dataset.zip`** from the Output panel.

## 3. Each review run (local / Mac)
1. Unzip `grove_dataset.zip` **into `./data`**. The zip's top-level folders are `work/` and
   `dataset/` (no `data/` prefix), so the `-d ./data` is required — a plain `unzip` would drop
   them in the wrong place and the review UI would find nothing:
   ```bash
   mkdir -p data && unzip -o grove_dataset.zip -d ./data
   ```
   Confirm `./data/work/manifest.json` and `./data/dataset/data.yaml` now exist (matching `config.yaml`).
2. Launch the review UI and correct boxes in the browser:
   ```bash
   make review
   ```
3. Fix every box, then click **Export** to write the final YOLO + COCO dataset to `./data/dataset`.
