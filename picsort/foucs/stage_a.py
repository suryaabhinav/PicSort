from typing import Any, Callable, Dict, List, Optional

import cv2
import pandas as pd

from picsort.config import AppConfig, Models, RuntimeContext
from picsort.detection.yolo_seg import YOLOProcessor
from picsort.foucs.metrics import masked_focus
from picsort.io.utils import load_bgr_exif_safe
from picsort.pipeline.orchestrator import log

ProgressFn = Optional[Callable[[int, int, Optional[str]], None]]


def stage_a(
    imgs: List,
    yolo_results: Dict[str, Dict[str, Any]],
    cfg: AppConfig,
    progress: ProgressFn = None,
) -> pd.DataFrame:
    """
    Calculate focus metrics for each image in the given list.

    Args:
        imgs (List): List of image paths.
        yolo_results (Dict[str, Dict[str, Any]]): YOLO results for each image.
        cfg (AppConfig, optional): AppConfig object containing configuration parameters. Defaults to AppConfig().
        progress (ProgressFn, optional): Progress callback function. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame containing focus metrics for each image.
    """

    rows = []
    log.info(f"[INFO] Calculating focus metrics...")
    for i, img in enumerate(imgs):
        if progress and (i % 5 == 0):
            progress(i, len(imgs), "focus-metrics")
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
                label = "blurry"
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

    return pd.DataFrame(rows)
