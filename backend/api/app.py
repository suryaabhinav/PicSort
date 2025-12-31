import asyncio
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

import numpy as np
from api.logging_config import get_logger, log
from api.progress import registry
from api.schema import StartRunRequest
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse
from picsort.config import AppConfig
from picsort.pipeline.orchestrator import (
    move_with_run_artifact,
    run_pipeline_background,
)
from sse_starlette.sse import EventSourceResponse

app = FastAPI(title="PicSort API", description="PicSort API", version="0.0.1")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle NumPy types."""

    def default(self, obj):
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


@app.exception_handler(Exception)
async def unhandled_exc_handler(request: Request, exc: Exception):
    log.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/api/runs/{run_id}/events")
async def stream_events(run_id: str):
    run = registry.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    async def event_stream():
        yield {"event": "hello", "data": json.dumps({"run_id": run_id})}
        while True:
            event = await run.events.get()
            yield {
                "event": event.get("event", "progress"),
                "data": json.dumps(event, cls=NumpyEncoder),
            }

    return EventSourceResponse(event_stream())


@app.get("/api/runs/{run_id}/status")
def status(run_id: str):
    run = registry.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    return {
        "run_id": run.id,
        "stage": run.stage,
        "progress": run.progress,
        "message": run.message,
        "error": run.error,
        "timings": run.timings,
        "paths": run.paths,
        "root": run.root,
    }


@app.post("/api/pipeline/start")
async def start_pipeline(request: StartRunRequest, bg_tasks: BackgroundTasks):
    run = registry.create()
    run.root = request.root
    run.loop = asyncio.get_running_loop()
    cfg = AppConfig(root=request.root)

    # Validate path immediately
    path_root = Path(request.root)
    if not path_root.exists() or not path_root.is_dir():
        raise HTTPException(
            status_code=400, detail=f"Path does not exist or is not a directory: {request.root}"
        )

    if request.batch_size:
        cfg.scene.batch_size = request.batch_size

    # Apply analysis parameters if provided
    if request.focus_t_subj is not None:
        cfg.focus.t_subj = request.focus_t_subj
    if request.focus_t_bg is not None:
        cfg.focus.t_bg = request.focus_t_bg
    if request.yolo_person_conf is not None:
        cfg.yolo.person_conf = request.yolo_person_conf
    if request.face_conf is not None:
        cfg.face.conf = request.face_conf
    if request.face_sim_tresh is not None:
        cfg.face.sim_tresh = request.face_sim_tresh

    # Setup run logger
    run_logger = get_logger(f"picsort.run.{run.id}")
    run_log_path = Path("runs") / run.id
    run_log_path.mkdir(parents=True, exist_ok=True)
    fh = RotatingFileHandler(
        run_log_path / "run.log", maxBytes=2_000_000, backupCount=2, encoding="utf-8"
    )
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    run_logger.addHandler(fh)
    run.run_logger = run_logger

    async def run_job():
        try:
            analytics = await asyncio.to_thread(
                run_pipeline_background, run, cfg, Path(request.root)
            )
            run.emit_threadsafe(event={"event": "result", "analytics": analytics})

        except Exception as e:
            log.exception("Pipeline failed for run %s", run.id)
            run.error = str(e)
            run.stage = "error"
            run.emit_threadsafe(event={"event": "error", "error": str(e)})

    task = asyncio.create_task(run_job())
    run.bg_task = task
    return {"run_id": run.id}


@app.post("/api/pipeline/stop/{run_id}")
async def stop_pipeline(run_id: str):
    run = registry.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    run.cancel_event.set()
    await run.emit(
        {"event": "stage", "stage": "stopping", "progress": run.progress, "msg": "Requested cancel"}
    )
    return {"run_id": run_id, "status": "cancelling"}


@app.post("/api/move/{run_id}")
async def move(run_id: str, dry_run: bool = False):
    run = registry.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    if not run.root or "final" not in run.paths:
        raise HTTPException(status_code=400, detail="Run not completed, or missing final artifact")

    result = await asyncio.to_thread(
        move_with_run_artifact, Path(run.root), Path(run.paths["final"]), dry_run
    )

    await run.emit({"event": "moved", "data": result})
    return {"run_id": run_id, "result": result}
