"""Grove command-line interface (CLAUDE.md §7).

Subcommands map one-to-one onto the pipeline stages:

    grove ingest    folder      -> manifest.json
    grove detect    manifest    -> predictions.json        (GPU)
    grove annotate  predictions -> preview images          (QC only)
    grove export    state       -> YOLO + COCO dataset      (the real deliverable)
    grove train     dataset     -> student YOLO weights     (GPU, optional)
    grove review    dataset     -> local FastAPI review UI

Every command reads the same single YAML config (``--config``, default
``config.yaml``) via :func:`grove.core.config.load_config`.

Why stage imports are LAZY (inside each command body, never at module top):
``grove review`` must run on the target Intel Mac that has installed ONLY the
``[review]`` extra — no cv2 / torch / autodistill / supervision / ultralytics.
If we imported the GPU-side stages at module import time, simply *loading* the
CLI to run ``review`` would crash with a missing heavy dependency. Importing
each stage only when its command actually runs keeps the CLI importable with
the light review-only install (and equally on the GPU side). See CLAUDE.md
§2.3, §10.
"""

from __future__ import annotations

import logging

import typer

from grove.core.config import load_config

# A sensible default so the user typically runs a bare ``grove <stage>``.
_DEFAULT_CONFIG = "config.yaml"

# Reused option definition: every subcommand accepts the same --config flag.
_ConfigOption = typer.Option(
    _DEFAULT_CONFIG,
    "--config",
    "-c",
    help="Path to the Grove YAML config file.",
    show_default=True,
)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Grove: open-vocabulary auto-labeling pipeline for mandarin detection datasets.",
)


def _setup_logging() -> None:
    """Configure stdlib logging once so stage modules' log calls are visible."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


@app.command()
def ingest(config: str = _ConfigOption) -> None:
    """Scan the input folder -> EXIF-normalized images + manifest.json."""
    _setup_logging()
    cfg = load_config(config)
    # Lazy import: ingest pulls in Pillow/cv2 which are absent on a review-only Mac.
    from grove.pipeline.ingest import ingest as _ingest

    manifest = _ingest(cfg)
    typer.echo(
        f"ingest: {len(manifest.images)} image(s) -> {cfg.paths.work_dir}/manifest.json"
    )


@app.command()
def detect(config: str = _ConfigOption) -> None:
    """Run the open-vocab detector over the manifest -> predictions.json (GPU)."""
    _setup_logging()
    cfg = load_config(config)
    # Lazy import: detect pulls in the GPU detector backend (autodistill/torch).
    from grove.pipeline.detect import detect as _detect

    manifest = _detect(cfg)
    n_boxes = sum(len(rec.detections) for rec in manifest.images)
    typer.echo(
        f"detect: {n_boxes} box(es) over {len(manifest.images)} image(s) "
        f"using backend '{cfg.detector.backend}' -> {cfg.paths.work_dir}/predictions.json"
    )


@app.command()
def annotate(config: str = _ConfigOption) -> None:
    """Draw boxes onto image copies for human QC -> work_dir/previews/."""
    _setup_logging()
    cfg = load_config(config)
    # Lazy import: annotate uses supervision/cv2 for drawing.
    from grove.pipeline.annotate import annotate as _annotate

    _annotate(cfg)
    typer.echo(f"annotate: previews written -> {cfg.paths.work_dir}/previews/")


@app.command()
def export(config: str = _ConfigOption) -> None:
    """Write the current annotation state as a YOLO + COCO dataset."""
    _setup_logging()
    cfg = load_config(config)
    # Lazy import: keeps heavy/optional deps out of the bare CLI load.
    from grove.pipeline.export import export as _export

    summary = _export(cfg)
    typer.echo(
        "export: {train} train / {val} val image(s), {boxes} box(es) -> {export_dir}".format(
            train=summary.get("train", "?"),
            val=summary.get("val", "?"),
            boxes=summary.get("boxes", "?"),
            export_dir=summary.get("export_dir", cfg.paths.export_dir),
        )
    )


@app.command()
def train(config: str = _ConfigOption) -> None:
    """Distill a student YOLO on the reviewed dataset (GPU, optional).

    Note: the trained YOLO student — not Grounding DINO — is the
    robot-deployable artifact (CLAUDE.md §2.1, §6.9).
    """
    _setup_logging()
    cfg = load_config(config)
    # Lazy import: train pulls in ultralytics/torch (GPU side only).
    from grove.pipeline.train import train as _train

    _train(cfg)
    typer.echo(f"train: student YOLO run complete -> {cfg.paths.work_dir}/runs/")


@app.command()
def review(config: str = _ConfigOption) -> None:
    """Launch the local FastAPI review UI to correct boxes (no GPU)."""
    _setup_logging()
    cfg = load_config(config)
    # Lazy import: keeps the GPU stages out of the review-only code path.
    from grove.review.server import run_server

    typer.echo(
        f"review: serving UI at http://{cfg.review.host}:{cfg.review.port}  (Ctrl-C to stop)"
    )
    run_server(cfg)


def main() -> None:
    """Console-script entry point (pyproject maps ``grove`` -> ``grove.cli:main``)."""
    app()


if __name__ == "__main__":
    main()
