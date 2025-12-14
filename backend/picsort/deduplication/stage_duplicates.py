from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from backend.api.logging_config import log
from backend.picsort.config import AppConfig
from backend.picsort.io.utils import list_images, md5_hash

ProgressFn = Optional[Callable[[int, int, Optional[str]], None]]


def stage_duplicates(root: Path, cfg: AppConfig, progress: ProgressFn = None) -> pd.DataFrame:
    """Stage Duplicates: Calculate MD5 hashes for all the images and find duplicates

    Args:
        root (Path): Root directory of images
        cfg (AppConfig): Configuration object
        progress (ProgressFn, optional): Progress callback function. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame of stage duplicates results
    """
    image_paths = list_images(root)
    rows = []
    for p in image_paths:
        path_key = p.name
        try:
            md5 = md5_hash(p)
            rows.append({"path": path_key, "md5": md5})
        except Exception as e:
            log.warning(f"Failed to calculate MD5 for {p}: {e}")
            continue

    df_stage_duplicates = pd.DataFrame(rows)
    df_stage_duplicates["is_duplicate"] = df_stage_duplicates.duplicated(subset="md5", keep="first")
    n_dupes = df_stage_duplicates["is_duplicate"].sum()

    if n_dupes > 0:
        log.info(f"Found {n_dupes} duplicates")
        log.info(f"Will process {len(df_stage_duplicates) - n_dupes} unique images")
    else:
        log.info("No duplicates found")

    return df_stage_duplicates


def broadcast_to_duplicates(
    df_final: pd.DataFrame, df_stage_duplicates: pd.DataFrame
) -> pd.DataFrame:
    """Broadcast final DataFrame to duplicate images based on md5 matching"""

    log.info(
        f"[INFO] Broadcasting results to {df_stage_duplicates["is_duplicate"].sum()} duplicate images."
    )

    md5_to_metrics = {}

    for _, row in df_final.iterrows():
        md5_hash = row["md5"]
        if pd.notna(md5_hash):
            metrics = {
                "subject_sharpness": row["subject_sharpness"],
                "background_sharpness": row["background_sharpness"],
                "num_faces_found": row["num_faces_found"],
                "person_count": row["person_count"],
                "focus_label": row["focus_label"],
                "mask_used": row["mask_used"],
                "scene_group": row["scene_group"],
                "identity_group": row["identity_group"],
                "_face_boxes": row["_face_boxes"],
                "_face_id_labels": row["_face_id_labels"],
                "final_group_name": row["final_group_name"],
            }
            md5_to_metrics[md5_hash] = metrics

    result_rows = []

    for _, row in df_stage_duplicates.iterrows():
        md5_hash = row["md5"]
        path = row["path"]
        result_row = {"path": path, "md5": md5_hash}

        if md5_hash in md5_to_metrics:
            result_row.update(md5_to_metrics[md5_hash])
        else:
            log.warning(f"[WARNING] No metrics found for md5 hash: {md5_hash} (path: {path})")
            result_row.update(
                {
                    "subject_sharpness": 0.0,
                    "background_sharpness": 0.0,
                    "num_faces_found": 0,
                    "person_count": 0,
                    "focus_label": "unknown",
                    "mask_used": "unknown",
                    "scene_group": -100,
                    "identity_group": -10,
                    "_face_boxes": [],
                    "_face_id_labels": [],
                    "final_group_name": "30_Outliers/02_Unclassified",
                }
            )

        result_rows.append(result_row)

    df_final_broadcast = pd.DataFrame(result_rows)
    log.info(f"[INFO] Final dataframe has {len(df_final_broadcast)} total images")

    return df_final_broadcast
