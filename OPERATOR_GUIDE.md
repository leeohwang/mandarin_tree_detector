# Grove — Operator Guide

Three things to do. Nothing else is manual.

## 1. One-time setup
1. Create a Kaggle account and PHONE-VERIFY it (Settings -> Phone Verification). This unlocks the free GPU and Internet; it cannot be scripted.
2. Clone this repo and run the one-command local setup:
   ```bash
   ./setup.sh
   ```

## 2. Each labeling run (Kaggle / GPU)
1. In the notebook's **Add Data** panel, attach your image folder (as a Dataset) and this Grove repo
   (as a Dataset — or let cell 2 `git clone` it). The image and repo paths are auto-detected; nothing to edit.
2. Open `notebooks/kaggle_label.ipynb`. In the right sidebar set **Accelerator -> GPU** and **Internet -> On**.
3. Click **Run All**. When it finishes, download **`grove_dataset.zip`** from the Output panel.

## 3. Each review run (local / Mac)
1. Unzip `grove_dataset.zip` into `./data` (so `./data/work` and `./data/dataset` exist, matching `config.yaml`).
2. Launch the review UI and correct boxes in the browser:
   ```bash
   make review
   ```
3. Fix every box, then click **Export** to write the final YOLO + COCO dataset to `./data/dataset`.
