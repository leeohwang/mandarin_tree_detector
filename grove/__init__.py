"""Grove: open-vocabulary auto-labeling pipeline for mandarin-detection datasets.

The package splits along the GPU boundary (see SPEC.md §3):

- ``grove.core``     — GPU-free shared core: data model, config, format converters.
- ``grove.pipeline`` — GPU side: ingest, detect, annotate, export, train.
- ``grove.review``   — local side: FastAPI + canvas review UI.

The two halves communicate only through files on disk (the dataset directory).
"""

__version__ = "0.1.0"
