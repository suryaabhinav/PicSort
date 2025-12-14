from pathlib import Path

import numpy as np
import pandas as pd

from backend.api.logging_config import log
from backend.picsort.config import AppConfig, Models, RuntimeContext
from backend.picsort.scene.hdbscan_wrap import cluster_hdbscan
from backend.picsort.scene.openclip import (embed_images_openclip_batch,
                                            get_openclip)


def stage_c(
    root: Path,
    df_stage_b: pd.DataFrame,
    cfg: AppConfig,
    ctx: RuntimeContext,
    models: Models,
    progress=None,
) -> pd.DataFrame:
    """Stage C: Cluster landscape images using OpenCLIP and HDBSCAN

    Args:
        root (Path): Root directory
        df_stage_b (pd.DataFrame): DataFrame from stage B
        cfg (AppConfig): App config
        ctx (RuntimeContext): Runtime context
        models (Models): Models
        progress (object, optional): Progress bar. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame with stage C results
    """

    df_stage_c = df_stage_b.copy()

    landscape_df = df_stage_c[
        (df_stage_c["focus_label"] == "subject_in_focus") & (df_stage_c["person_count"] == 0)
    ].copy()
    landscape_paths = [root / row["path"] for _, row in landscape_df.iterrows()]
    if not landscape_paths:
        log.info(f"[INFO] Stage C: No landscape images to process")
        return df_stage_c

    total = len(landscape_paths)
    log.info(f"[INFO] Stage C: Processing {total} landscape images")

    if models.openclip_model is None or models.openclip_preprocess is None:
        models.openclip_model, models.openclip_preprocess = get_openclip(ctx)

    embeds = embed_images_openclip_batch(
        model=models.openclip_model,
        preprocess=models.openclip_preprocess,
        paths=landscape_paths,
        cfg=cfg,
        ctx=ctx,
    )

    if not embeds:
        return df_stage_c

    names = list(embeds.keys())
    E = np.vstack([embeds[n] for n in names]).astype(np.float64, copy=False)
    labels = cluster_hdbscan(E, cfg)

    for name, label in zip(names, labels):
        mask = df_stage_c["path"] == name
        df_stage_c.loc[mask, "scene_group"] = int(label)

    log.info(f"[INFO] Stage C: Found {len(df_stage_c["scene_group"].unique())} unique scene groups")
    return df_stage_c
