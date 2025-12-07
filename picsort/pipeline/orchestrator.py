import logging
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from picsort.config import AppConfig, Models, RuntimeContext
from picsort.deduplication.stage_duplicates import stage_duplicates
from picsort.detection.yolo_seg import YOLOProcessor
from picsort.faces.stage_b import stage_b
from picsort.foucs.stage_a import stage_a
from picsort.grouping.final import grouping
from picsort.io.utils import list_images
from picsort.scene.stage_c import stage_c


def make_logger(script_path: Path) -> logging.Logger:
    log = logging.getLogger("picsort")
    log.setLevel(logging.DEBUG)
    log.propagate = False
    if not log.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        log.addHandler(ch)
        fh = logging.FileHandler(script_path.parent / "picsort_pipeline.log", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s")
        )
        log.addHandler(fh)
    return log


SCRIPT_PATH = Path(__file__).resolve()
log = make_logger(SCRIPT_PATH)


def run_stage_duplicates(root: Path, cfg: AppConfig, progress=None) -> pd.DataFrame:
    return stage_duplicates(root, cfg, progress)


def run_stage_a(
    root: Path,
    df_stage_duplicates: pd.DataFrame,
    cfg: AppConfig,
    ctx: RuntimeContext,
    models: Models,
    progress: None,
) -> pd.DataFrame:

    unique_paths_set = set(
        df_stage_duplicates[~df_stage_duplicates["is_duplicate"]]["path"].tolist()
    )
    imgs_to_process = [p for p in list_images(root) if p.name in unique_paths_set]

    log.info(f"[INFO] Stage A: Processing {len(imgs_to_process)} unique images")

    yolo = YOLOProcessor(cfg, ctx, models)
    yolo_results = yolo.process_batch(imgs_to_process)

    df_stage_a = stage_a(imgs_to_process, yolo_results, cfg, progress)

    df_stage_a = df_stage_a.merge(df_stage_duplicates["path", "md5"], on="path", how="left")

    return df_stage_a


def run_stage_b_faces(
    root: Path,
    df_stage_a: pd.DataFrame,
    cfg: AppConfig,
    ctx: RuntimeContext,
    models: Models,
    progress=None,
) -> pd.DataFrame:

    df_stage_b = stage_b(root, df_stage_a, cfg, ctx, models, progress)

    return df_stage_b


def run_stage_c_scene(
    root: Path,
    df_stage_b: pd.DataFrame,
    cfg: AppConfig,
    ctx: RuntimeContext,
    models: Models,
    progress=None,
) -> pd.DataFrame:

    df_stage_c = stage_c(root, df_stage_b, cfg, ctx, models, progress)

    return df_stage_c


def apply_grouping(df_stage_c: pd.DataFrame) -> Dict[str, Any]:

    df_final = grouping(df_stage_c)

    return {"moved": 0, "skipped": 0, "dry_run": dry_run}
