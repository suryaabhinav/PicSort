from pathlib import Path

import pandas as pd

from backend.picsort.config import AppConfig, Models, RuntimeContext
from backend.picsort.deduplication.stage_duplicates import (
    broadcast_to_duplicates, stage_duplicates)
from backend.picsort.faces.stage_b import stage_b
from backend.picsort.foucs.stage_a import stage_a
from backend.picsort.grouping.final import grouping
from backend.picsort.scene.stage_c import stage_c


def run_stage_duplicates(root: Path, cfg: AppConfig, progress=None) -> pd.DataFrame:
    df_stage_duplicates = stage_duplicates(root, cfg, progress)
    return df_stage_duplicates


def run_stage_a(
    root: Path,
    df_stage_duplicates: pd.DataFrame,
    cfg: AppConfig,
    ctx: RuntimeContext,
    models: Models,
    progress: None,
) -> pd.DataFrame:

    df_stage_a = stage_a(root, df_stage_duplicates, ctx, models, cfg, progress)

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


def apply_grouping(df_stage_c: pd.DataFrame) -> pd.DataFrame:

    df_final = grouping(df_stage_c)

    return df_final


def broadcast(df_final: pd.DataFrame, df_stage_duplicates: pd.DataFrame) -> pd.DataFrame:

    df_final_broadcast = broadcast_to_duplicates(df_final, df_stage_duplicates)

    return df_final_broadcast


def move(root: Path, df_final: pd.DataFrame) -> None:
    move_images(root, df_final, path_col="path", group_col="final_group_name")
    return {"moved_plan": df_final["final_group_name"].value_counts().to_dict()}
