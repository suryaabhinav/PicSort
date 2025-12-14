from pathlib import Path
from typing import Callable, Optional

import cv2
import pandas as pd

from api.logging_config import log
from picsort.config import AppConfig, Models, RuntimeContext
from picsort.detection.yolo_seg import YOLOProcessor
from picsort.foucs.metrics import masked_focus
from picsort.io.utils import list_images, load_bgr_exif_safe

ProgressFn = Optional[Callable[[int, int, Optional[str]], None]]


def stage_a(
    root: Path,
    df_stage_duplicates: pd.DataFrame,
    ctx: RuntimeContext,
    models: Models,
    cfg: AppConfig,
    progress: ProgressFn = None,
) -> pd.DataFrame:
    """Stage A: Focus Metrics

    Args:
        root (Path): Root directory of the images
        df_stage_duplicates (pd.DataFrame): DataFrame containing the duplicates
        ctx (RuntimeContext): Runtime context
        models (Models): Models
        cfg (AppConfig): AppConfig
        progress (ProgressFn, optional): Progress callback function. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame containing the focus metrics
    """

    unique_paths_set = set(
        df_stage_duplicates[~df_stage_duplicates["is_duplicate"]]["path"].tolist()
    )
    imgs_to_process = [p for p in list_images(root) if p.name in unique_paths_set]
    log.info(f"[INFO] Stage A: Processing {len(imgs_to_process)} unique images")

    yolo = YOLOProcessor(cfg, ctx, models)
    yolo_results = yolo.process_batch(imgs_to_process, cfg.yolo.batch_size)

    rows = []
    log.info(f"[INFO] Calculating focus metrics...")
    for i, img in enumerate(imgs_to_process):
        try:
            if progress and (i % 5 == 0):
                progress(i, len(imgs_to_process), "focus-metrics")
            bgr = load_bgr_exif_safe(img)
            if bgr is None:
                continue

            path_key = img.name
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

            yolo_data = yolo_results.get(path_key, None)
            if yolo_data is None:
                continue

            if yolo_data is None or yolo_data.get("mask") is None:
                mask = YOLOProcessor._ellipse(bgr)
                yolo_data = {"mask": mask, "person_count": 0, "mask_type": "ellipse"}
            else:
                mask = yolo_data["mask"]

            bg_mask = ~mask

            subj_s = masked_focus(gray, mask)
            bg_s = masked_focus(gray, bg_mask)

            t_subj = cfg.focus.t_subj
            t_bg = cfg.focus.t_bg
            label = "subject_in_focus"
            if subj_s < t_subj:
                if bg_s < (t_subj / 2):
                    label = "overall_soft_low_quality"
                else:
                    label = "subject_soft_general"
            elif bg_s > t_bg and abs(subj_s - bg_s) < cfg.focus.closeness:
                label = "overall_sharp_flat_focus"

            rows.append(
                {
                    "path": path_key,
                    "subject_sharpness": round(subj_s, 2),
                    "background_sharpness": round(bg_s, 2),
                    "num_faces_found": 0,
                    "person_count": int(yolo_data["person_count"]),
                    "focus_label": label,
                    "mask_used": yolo_data["mask_type"],
                    "scene_group": -100,
                    "identity_group": -10,
                    "_face_boxes": [],
                    "_face_id_labels": [],
                }
            )
        except Exception as e:
            log.warning(f"[WARNING] Failed to calculate focus metrics for {img}: {e}")
            rows.append(
                {
                    "path": path_key,
                    "subject_sharpness": 0.0,
                    "background_sharpness": 0.0,
                    "num_faces_found": 0,
                    "person_count": 0,
                    "focus_label": "Unknown",
                    "mask_used": "Unknown",
                    "scene_group": -100,
                    "identity_group": -10,
                    "_face_boxes": [],
                    "_face_id_labels": [],
                }
            )
    df_stage_a = pd.DataFrame(rows)

    try:
        df_stage_a = df_stage_a.merge(df_stage_duplicates[["path", "md5"]], on="path", how="left")
    except Exception as e:
        log.warning(f"[WARNING] Failed to merge stage A and stage duplicates: {e}")

    return df_stage_a
