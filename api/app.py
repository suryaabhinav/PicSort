import asyncio

from fastapi import BackgroundTasks, FastAPI
from fastapi.exceptions import HTTPException
from fastapi.param_functions import Path
from sse_starlette import EventSourceResponse

from api.progress import registry
from api.schema import StartRunRequest
from picsort.config import AppConfig, Models, RuntimeContext
from picsort.pipeline.orchestrator import (
    apply_grouping,
    run_stage_a,
    run_stage_b_faces,
    run_stage_c_scene,
    run_stage_duplicates,
)

app = FastAPI(title="PicSort API", description="PicSort API", version="0.0.1")


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/api/runs/{run_id}/events")
async def stream_events(run_id: str):
    run = registry.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    async def event_stream():
        yield {"event": "hello", "data": f"{run_id}"}
        while True:
            event = await run.events.get()
            yield {"event": event.get("event", "progress"), "data": event}

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
    }


@app.post("/api/pipeline/start")
async def start_pipeline(request: StartRunRequest, bg_tasks: BackgroundTasks):
    run = registry.create()

    async def run_job():
        try:
            cfg = AppConfig(root=request.root)
            ctx = RuntimeContext()
            models = Models()
            if request.batch_size:
                cfg.yolo.batch_size = request.batch_size

            # Stage Duplicates
            run.stage, run.progress, run.message = (
                "stage_duplicates",
                0.0,
                "Starting stage duplicates",
            )
            await run.emit(
                {"event": "stage", "stage": "stage_duplicates", "progress": 0.0, "msg": "Begin"}
            )

            df_stage_duplicates = await asyncio.to_thread(
                run_stage_duplicates,
                Path(request.root),
                cfg,
                progress=lambda i, t, msg=None: asyncio.create_task(
                    run.emit(
                        {
                            "event": "progress",
                            "stage": "stage_duplicates",
                            "progress": i / max(t, 1),
                            "msg": msg or "",
                        }
                    )
                ),
            )

            # Stage A
            run.stage, run.progress, run.message = "stage_a", 0.0, "Starting stage A"
            await run.emit({"event": "stage", "stage": "stage_a", "progress": 0.0, "msg": "Begin"})

            df_stage_a = await asyncio.to_thread(
                run_stage_a,
                Path(request.root),
                df_stage_duplicates,
                cfg,
                ctx,
                models,
                progress=lambda i, t, msg=None: asyncio.create_task(
                    run.emit(
                        {
                            "event": "progress",
                            "stage": "stage_a",
                            "progress": i / max(t, 1),
                            "msg": msg or "",
                        }
                    )
                ),
            )

            # Stage B
            run.stage, run.progress, run.message = "stage_a", 1.0, "Completed stage A"
            await run.emit(
                {"event": "stage", "stage": "stage_a", "progress": 1.0, "msg": "Completed"}
            )

            if request.run_faces:
                run.stage, run.progress, run.message = "faces", 0.0, "Starting stage B (faces)"
                await run.emit(
                    {"event": "stage", "stage": "faces", "progress": 0.0, "msg": "Begin"}
                )

                df_stage_b = await asyncio.to_thread(
                    run_stage_b_faces,
                    Path(request.root),
                    df_stage_a,
                    cfg,
                    ctx,
                    models
                    progress=lambda i, t, msg=None: asyncio.create_task(
                        run.emit(
                            {
                                "event": "progress",
                                "stage": "faces",
                                "progress": i / max(t, 1),
                                "msg": msg or "",
                            }
                        )
                    ),
                )

                run.stage, run.progress, run.message = "faces", 1.0, "Completed stage B (faces)"
                await run.emit(
                    {"event": "stage", "stage": "faces", "progress": 1.0, "msg": "Completed"}
                )

            else:
                df_stage_b = None

            # Stage C: Scene Clustring
            run.stage, run.progress, run.message = "scene clustring", 0.0, "Clustring landscape scenes"
            await run.emit({"event": "stage", "stage": "scene clustring", "progress": 0.0, "msg": "Begin"})

            df_stage_c = await asyncio.to_thread(
                run_stage_c_scene,
                Path(request.root),
                df_stage_b,
                cfg,
                ctx,
                models,
                progress=lambda i, t, msg=None: asyncio.create_task(
                    run.emit(
                        {
                            "event": "progress",
                            "stage": "scene clustring",
                            "progress": i / max(t, 1),
                            "msg": msg or ""
                        }
                    )
                )
            )
            await run.emit(
                {"events": "stage", "stage": "scene clustring", "progress": 1.0, "msg": "Completed"}
            )

            # Stage Apply Grouping
            run.stage, run.progress, run.message = "grouping", 0.0, "Applying grouping"
            await run.emit({"event": "stage", "stage":"grouping","progress": 0.0, "msg": "Begin"})
            

            result = await asyncio.to_thread(
                apply_grouping, df_stage_c,
            )
            await run.emit({"event": "stage", "stage": "grouping", "progress": 1.0, "msg": "Completed"})

            await run.emit({"event": "done", "ok": True, "result": result})

        except Exception as e:
            run.error = str(e)
            run.stage = "error"
            await run.emit({"event": "error", "error": str(e)})

    bg_tasks.add_task(run_job)
    return {"run_id": run.id}
