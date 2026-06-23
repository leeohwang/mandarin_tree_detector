"""Optional student-distillation stage — train a fast YOLO on REVIEWED labels.

Read CLAUDE.md §6.9, §2 (point 1), and §12 before changing anything here.

THE TEACHER vs STUDENT DISTINCTION IS THE WHOLE POINT OF THIS FILE — keep it loud:

- Grounding DINO (the open-vocab detector used in `grove detect`) is the *teacher*.
  It is accurate but SLOW and is NOT deployable on the robot — never treat it as a
  runtime detector (CLAUDE.md §2 point 1, §12 "Deploying the teacher").
- This stage distills the teacher's HUMAN-REVIEWED labels into a *student* YOLO
  (ultralytics). THOSE STUDENT WEIGHTS — not Grounding DINO — are the only
  robot-deployable artifact produced by this whole pipeline.
- A student is only as good as the labels it learns from, so we train on the
  EXPORTED dataset, which should be the *reviewed* corrections, not raw auto-labels
  (CLAUDE.md §2 point 2, §12 "Treating auto-labels as final"). We don't re-validate
  that here, but we log a loud reminder.

GPU side (Kaggle). Import discipline: ultralytics is a heavy GPU dependency, so it
is imported LAZILY inside train() — this module must be importable by the CLI on a
Mac that only has the [review] extra (CLAUDE.md §10, core import discipline).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from grove.core.config import Config

logger = logging.getLogger(__name__)


def train(cfg: Config) -> None:
    """Distill a student YOLO on the reviewed, exported dataset (CLAUDE.md §6.9).

    Trains an ultralytics ``YOLO(cfg.train.model)`` on ``<export_dir>/data.yaml``
    for ``cfg.train.epochs`` epochs at ``imgsz=cfg.train.imgsz``, saving runs under
    ``<work_dir>/runs/``. No-ops (with a log line) when ``cfg.train.enabled`` is
    False so a "run everything" notebook can call it unconditionally.
    """
    # --- Guard: training is an OPTIONAL stage, off by default (§6.2, §6.9). ------
    if not cfg.train.enabled:
        logger.info(
            "Training is disabled (train.enabled=false) — skipping student "
            "distillation. Set train.enabled: true in the config to train a "
            "deployable YOLO student on the reviewed dataset."
        )
        return

    work_dir = Path(cfg.paths.work_dir)
    export_dir = Path(cfg.paths.export_dir)
    data_yaml = export_dir / "data.yaml"

    # The student learns ONLY from what export wrote. If the dataset isn't there,
    # fail loudly with the fix rather than letting ultralytics emit a cryptic error.
    if not data_yaml.is_file():
        raise FileNotFoundError(
            f"No dataset found at {data_yaml}. Run `grove export` first (ideally "
            "after reviewing the auto-labels) so the student trains on corrected "
            "labels, not raw teacher output (CLAUDE.md §2, §12)."
        )

    # runs live under work_dir so they travel with the rest of the run's artifacts.
    # MUST be absolute: ultralytics re-roots a RELATIVE `project` under its own
    # settings runs_dir (e.g. ./runs/detect/<project>), scattering weights outside
    # work_dir. An absolute project makes save_dir exactly <work_dir>/runs/<name>.
    runs_dir = (work_dir / "runs").resolve()
    runs_dir.mkdir(parents=True, exist_ok=True)
    # Timestamped, project-scoped run name so repeated trainings don't clobber.
    run_name = f"{cfg.project_name}-student-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"

    # LOUD reminder of what we're producing and why review matters (§2 point 2).
    logger.info(
        "Distilling TEACHER (Grounding DINO) labels into a STUDENT YOLO. "
        "The student is the robot-deployable artifact; Grounding DINO is NOT "
        "(CLAUDE.md §2, §12). Training assumes %s holds REVIEWED labels.",
        data_yaml,
    )
    logger.info(
        "Training student '%s' on %s for %d epochs at imgsz=%d (runs -> %s/%s)",
        cfg.train.model,
        data_yaml,
        cfg.train.epochs,
        cfg.train.imgsz,
        runs_dir,
        run_name,
    )

    # --- Lazy heavy import: keeps this module importable without ultralytics. ----
    # (CLI on the [review]-only Mac must be able to import grove.pipeline.train.)
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - depends on optional [gpu] extra
        raise ImportError(
            "ultralytics is required for `grove train` but is not installed. "
            "Install the GPU extra (e.g. `pip install -e .[gpu]`) on the Kaggle/"
            "GPU machine. Training is a GPU-side stage (CLAUDE.md §6.9, §10)."
        ) from exc

    # cfg.train.model may be a checkpoint name (e.g. "yolov8n.pt"); ultralytics
    # auto-downloads pretrained weights on first use — no manual fetch (§10.4).
    model = YOLO(cfg.train.model)

    results = model.train(
        data=str(data_yaml),
        epochs=cfg.train.epochs,
        imgsz=cfg.train.imgsz,
        project=str(runs_dir),
        name=run_name,
    )

    # ultralytics writes everything under <project>/<name>/. Surface exactly where
    # the deployable weights and validation metrics landed so the operator (and any
    # downstream robot-deployment step) can find them without guessing.
    save_dir = Path(getattr(results, "save_dir", runs_dir / run_name))
    best_weights = save_dir / "weights" / "best.pt"

    logger.info(
        "Student training complete. DEPLOYABLE ARTIFACT = these YOLO student "
        "weights, NOT Grounding DINO (CLAUDE.md §2 point 1, §12). "
        "Best weights: %s",
        best_weights,
    )
    logger.info(
        "Validation metrics + plots saved under %s (see results.csv, *.png).",
        save_dir,
    )
