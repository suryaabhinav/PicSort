import time
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
from api.logging_config import get_logger
from picsort.config import AppConfig, Models, RuntimeContext
from picsort.pipeline.orchestrator_core import (
    apply_grouping,
    broadcast,
    run_stage_a,
    run_stage_b_faces,
    run_stage_c_scene,
    run_stage_duplicates,
)
from picsort.utils.device import choose_device
from picsort.utils.helpers import apply_move

log = get_logger()


def _now() -> float:
    """Returns the current time in seconds."""
    return time.perf_counter()


def init_runtime(cfg: AppConfig) -> Tuple[RuntimeContext, Models]:
    """Initializes the runtime context and models."""
    ctx = choose_device(cfg.yolo.device)
    models = Models()
    return ctx, models


def build_analytics(df_final: pd.DataFrame) -> Dict[str, Any]:
    """Builds analytics from the final DataFrame

    Args:
        df_final (pd.DataFrame): Final DataFrame

    Returns:
        Dict[str, Any]: Analytics dictionary
    """

    def safe_counts(series: pd.Series) -> Dict[str, int]:
        return {str(k): int(v) for k, v in series.value_counts(dropna=False).items()}

    analytics: Dict[str, Any] = {
        "total": {
            "images": int(len(df_final)),
            "with_people": int((df_final["person_count"] > 0).sum()),
            "without_people": int((df_final["person_count"] == 0).sum()),
        },
        "focus_distribution": safe_counts(df_final["focus_label"]),
        "person_count_distribution": safe_counts(df_final["person_count"]),
        "scene_groups": safe_counts(df_final.get("scene_group", pd.Series([-1] * len(df_final)))),
        "final_groups": safe_counts(df_final["final_group_name"]),
        "subject_sharpness_hist": (
            list(
                pd.cut(df_final["subject_sharpness"], bins=10, labels=False)
                .value_counts()
                .sort_index()
                .values
            )
            if "subject_sharpness" in df_final
            else []
        ),
        "cluster_sizes": [len(cluster) for cluster in df_final["final_group_name"].unique()],
    }

    return analytics


def run_pipeline_background(run_state, cfg: AppConfig, root: Path) -> Dict[str, Any]:
    """Runs the pipeline in the background.

    Args:
        run_state (RunState): Run state object
        cfg (AppConfig): App configuration
        root (Path): Root directory

    Returns:
        Dict[str, Any]: Analytics dictionary
    """

    def emit(event: Dict[str, Any]):
        run_state.emit_threadsafe(event)

    def stage(name: str, msg: str, p: float = 0.0):
        run_state.stage, run_state.message, run_state.progress = name, msg, p
        emit({"event": "stage", "stage": name, "progress": p, "msg": msg})

    def check_cancel():
        if run_state.cancel_event.is_set():
            run_state.stage = "cancelled"
            emit({"event": "cancelled", "msg": "Run cancelled"})
            raise RuntimeError("Run cancelled")

    t0 = _now()
    ctx, models = init_runtime(cfg)

    # Setup Local Logger from run_state if available
    logger = getattr(run_state, "run_logger", log)

    # Stage Duplicates
    try:
        stage("stage_duplicates", "Starting stage duplicates", 0.0)
        df_stage_duplicates = run_stage_duplicates(
            root,
            cfg,
            progress=lambda i, t, msg=None: emit(
                {
                    "event": "progress",
                    "stage": "stage_duplicates",
                    "progress": i / max(t, 1),
                    "msg": msg or "",
                }
            ),
        )
        run_state.timings["stage_duplicates_s"] = _now() - t0
        stage("stage_duplicates", "Completed stage duplicates", 1.0)
        check_cancel()
    except Exception:
        logger.exception("Stage 'duplicates' failed (run=%s)", run_state.id)
        raise

    # Stage A
    try:
        stage("stage_a", "Starting stage A", 0.0)
        ta = _now()
        df_stage_a = run_stage_a(
            root,
            df_stage_duplicates,
            cfg,
            ctx,
            models,
            progress=lambda i, t, msg=None: emit(
                {
                    "event": "progress",
                    "stage": "stage_a",
                    "progress": i / max(t, 1),
                    "msg": msg or "",
                }
            ),
        )
        run_state.timings["stage_a_s"] = _now() - ta
        stage("stage_a", "Completed stage A", 1.0)
        check_cancel()
    except Exception:
        logger.exception("Stage 'A' failed (run=%s)", run_state.id)
        raise

    # Stage B
    try:
        stage("stage_b", "Starting stage B", 0.0)
        tb = _now()
        df_stage_b = run_stage_b_faces(
            root,
            df_stage_a,
            cfg,
            ctx,
            models,
            progress=lambda i, t, msg=None: emit(
                {
                    "event": "progress",
                    "stage": "stage_b",
                    "progress": i / max(t, 1),
                    "msg": msg or "",
                }
            ),
        )
        run_state.timings["stage_b_s"] = _now() - tb
        stage("stage_b", "Completed stage B", 1.0)
        check_cancel()
    except Exception:
        logger.exception("Stage 'B' failed (run=%s)", run_state.id)
        raise

    # Stage C
    try:
        stage("stage_c", "Starting stage C", 0.0)
        tc = _now()
        df_stage_c = run_stage_c_scene(
            root,
            df_stage_b,
            cfg,
            ctx,
            models,
            progress=lambda i, t, msg=None: emit(
                {
                    "event": "progress",
                    "stage": "stage_c",
                    "progress": i / max(t, 1),
                    "msg": msg or "",
                }
            ),
        )
        run_state.timings["stage_c_s"] = _now() - tc
        stage("stage_c", "Completed stage C", 1.0)
        check_cancel()
    except Exception:
        logger.exception("Stage 'C' failed (run=%s)", run_state.id)
        raise

    # Stage Grouping
    try:
        stage("grouping", "Starting stage grouping", 0.0)
        tg = _now()
        df_stage_grouping = apply_grouping(df_stage_c)
        run_state.timings["stage_grouping_s"] = _now() - tg
        stage("grouping", "Completed stage grouping", 1.0)
        check_cancel()
    except Exception:
        logger.exception("Stage 'grouping' failed (run=%s)", run_state.id)
        raise

    # Final
    try:
        stage("finalize", "Starting final", 0.0)
        tf = _now()
        df_final = broadcast(df_stage_grouping, df_stage_duplicates)
        run_state.timings["finalize_s"] = _now() - tf
        stage("finalize", "Completed final", 1.0)
    except Exception:
        logger.exception("Stage 'finalize' failed (run=%s)", run_state.id)
        raise

    # Save artifacts
    out_dir = Path("runs") / run_state.id
    out_dir.mkdir(parents=True, exist_ok=True)
    p_a = out_dir / "stage_a.parquet"
    df_stage_a.to_parquet(p_a)
    run_state.paths["stage_a"] = str(p_a)
    p_b = out_dir / "stage_b.parquet"
    df_stage_b.to_parquet(p_b)
    run_state.paths["stage_b"] = str(p_b)
    p_c = out_dir / "stage_c.parquet"
    df_stage_c.to_parquet(p_c)
    run_state.paths["stage_c"] = str(p_c)
    p_g = out_dir / "grouping.parquet"
    df_stage_grouping.to_parquet(p_g)
    run_state.paths["grouping"] = str(p_g)
    p_f = out_dir / "final.parquet"
    df_final.to_parquet(p_f)
    run_state.paths["final"] = str(p_f)

    # Analytics JSON
    analytics = build_analytics(df_final)
    emit({"event": "analytics", "data": analytics})

    run_state.stage, run_state.progress, run_state.message = "done", 1.0, "All done"
    emit({"event": "done", "ok": True, "analytics": analytics})
    run_state.timings["total_s"] = _now() - t0

    return analytics


def move_with_run_artifact(root: Path, final_parquet: Path, dry_run: bool) -> Dict[str, Any]:
    df_final = pd.read_parquet(final_parquet)
    return apply_move(root, df_final, dry_run=dry_run)
