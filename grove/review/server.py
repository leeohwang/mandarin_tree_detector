"""FastAPI review server — the local, GPU-free correction loop (SPEC.md §6.8).

This is the HTTP layer that fronts the AnnotationStore. It runs locally on the
Mac with only the ``[review]`` extra installed (SPEC.md §10 local side), so the
import discipline here is strict: FastAPI / Starlette / uvicorn + std lib + grove
core + grove.review.store + grove.pipeline.export.write_dataset ONLY. No torch,
cv2, supervision, or ultralytics — none of the GPU/heavy stack — may be imported
from this module, directly or transitively (write_dataset is itself light).

Server-authoritative state (SPEC.md §12): there is exactly ONE AnnotationStore
instance per app, created at startup, and it is the single source of truth for the
reviewer's corrections. The frontend never trusts browser storage; every edit is a
round-trip to this store, which persists each change to <work_dir>/review_store.json.

Coordinate convention (SPEC.md §8/§9): every box crossing this API is CANONICAL
— normalized xyxy, top-left origin, each value in [0, 1]. The store already speaks
canonical; the canvas UI converts pixel<->canonical at its own boundary. Nothing
here rescales coordinates.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from grove.core.config import Config
from grove.pipeline.export import write_dataset
from grove.review.store import AnnotationStore

logger = logging.getLogger(__name__)

# The vanilla HTML/CSS/JS canvas UI lives alongside this module. Resolve it
# relative to this file so it is found regardless of the process CWD.
_STATIC_DIR = Path(__file__).parent / "static"

# Map file extension -> media type for GET /api/images/{id}/file. We avoid the
# stdlib mimetypes module's platform variance and just enumerate the formats
# ingest is allowed to produce (.jpg/.jpeg/.png/.webp, SPEC.md §6.3).
_MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


# -----------------------------------------------------------------------------
# Request body schemas. Pydantic validates shape/type at the HTTP boundary so a
# malformed payload yields a 422 rather than reaching the store as garbage.
# -----------------------------------------------------------------------------
class BoxIn(BaseModel):
    """One canonical box from the canvas UI. score is optional — human-drawn
    boxes have no detector confidence (the store will store it as None)."""

    label: str
    x1: float
    y1: float
    x2: float
    y2: float
    score: float | None = None


class BoxesIn(BaseModel):
    """PUT /api/images/{id}/boxes body: the full replacement box set for the
    image (the UI always sends the complete corrected list, never a delta)."""

    boxes: list[BoxIn]


class StatusIn(BaseModel):
    """POST /api/images/{id}/status body."""

    status: str


def create_app(cfg: Config) -> FastAPI:
    """Build the FastAPI app with one shared, authoritative AnnotationStore.

    The store is instantiated ONCE here (server-side state, §12). All endpoints
    close over this single instance, so every reviewer action mutates and
    persists the same working manifest.
    """
    store = AnnotationStore(cfg)
    app = FastAPI(title="Grove Review")

    # --- project metadata + progress ----------------------------------------
    @app.get("/api/meta")
    def get_meta() -> dict:
        return store.meta()

    # --- review queue --------------------------------------------------------
    @app.get("/api/images")
    def list_images() -> dict:
        return store.list_images()

    # --- single image: metadata + canonical boxes ---------------------------
    @app.get("/api/images/{image_id}")
    def get_image(image_id: str) -> dict:
        data = store.get_image(image_id)
        if data is None:
            raise HTTPException(status_code=404, detail=f"unknown image id {image_id!r}")
        return data

    # --- single image: raw prepared-image bytes -----------------------------
    @app.get("/api/images/{image_id}/file")
    def get_image_file(image_id: str) -> FileResponse:
        path = store.image_path(image_id)
        if path is None:
            raise HTTPException(status_code=404, detail=f"unknown image id {image_id!r}")
        if not path.exists():
            # The id is known but the prepared file is missing on disk — a real
            # operator error (e.g. work_dir/images was not transferred).
            raise HTTPException(
                status_code=404, detail=f"image file missing on disk for id {image_id!r}"
            )
        media_type = _MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")
        return FileResponse(path, media_type=media_type)

    # --- replace this image's boxes with the corrected set ------------------
    @app.put("/api/images/{image_id}/boxes")
    def put_boxes(image_id: str, body: BoxesIn) -> dict:
        # Hand the store plain dicts in its expected canonical shape. The store
        # funnels each through BBox (clamp + reject degenerate) and persists.
        boxes = [b.model_dump() for b in body.boxes]
        ok = store.set_boxes(image_id, boxes)
        if not ok:
            raise HTTPException(status_code=404, detail=f"unknown image id {image_id!r}")
        return {"ok": True}

    # --- mark reviewed / skipped / pending ----------------------------------
    @app.post("/api/images/{image_id}/status")
    def post_status(image_id: str, body: StatusIn) -> dict:
        try:
            ok = store.set_status(image_id, body.status)
        except ValueError as exc:
            # Bad status string -> 400 (the store validates the enum for us).
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not ok:
            raise HTTPException(status_code=404, detail=f"unknown image id {image_id!r}")
        return {"ok": True}

    # --- re-export the final dataset from corrected state -------------------
    @app.post("/api/export")
    def post_export() -> dict:
        # Re-export the REAL deliverable (coordinate files, §1) from the current
        # corrected working manifest. One shared writer (export.write_dataset)
        # owns the on-disk layout + validation, here and in the CLI.
        summary = write_dataset(
            store.to_manifest(),
            cfg.classes,
            cfg.paths.work_dir,
            cfg.paths.export_dir,
            cfg.export.formats,
            cfg.export.val_split,
            cfg.export.seed,
        )
        return {"ok": True, "summary": summary}

    # Static canvas UI mounted LAST, at "/", so the /api routes above take
    # precedence and "/" serves index.html (html=True). Registering this before
    # the API routes would let the catch-all swallow /api/* requests.
    app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="static")

    return app


def run_server(cfg: Config) -> None:
    """Launch the review UI with uvicorn on the configured host/port (§6.8).

    Logs the URL the operator should open — the only manual action on the local
    side beyond correcting boxes (OPERATOR_GUIDE, §10.4).
    """
    import uvicorn  # lazy: keep module import cheap and uvicorn optional at import time

    host, port = cfg.review.host, cfg.review.port
    logger.info("Grove review UI running at http://%s:%d  (open this in your browser)", host, port)
    uvicorn.run(create_app(cfg), host=host, port=port)
