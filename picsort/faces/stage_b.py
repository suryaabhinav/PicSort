import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from api.logging_config import get_logger, log
from picsort.config import AppConfig, Models, RuntimeContext
from picsort.detection.retina_mtcnn import (
    build_mtcnn,
    build_retinaface,
    detect_faces_smart,
)
from picsort.detection.yolo_face import build_yolov8_face
from picsort.faces.clustering import graph_clusters, remap_labels_sequential
from picsort.faces.embeddings import embed_face_batch, get_facenet_embedder
from picsort.io.utils import load_bgr_exif_safe
from picsort.utils.helpers import to_box_list_strict

ProgressFn = Optional[Callable[[int, int, Optional[str]], None]]


def stage_b(
    root: Path,
    df_stage_a: pd.DataFrame,
    cfg: AppConfig,
    ctx: RuntimeContext,
    models: Models,
    progress: ProgressFn = None,
) -> pd.DataFrame:
    """Stage B: Face detection and identity clustering

    Args:
        root (Path): Root directory of images
        df_stage_a (pd.DataFrame): DataFrame of stage A results
        cfg (AppConfig): Configuration object
        ctx (RuntimeContext): Runtime context
        models (Models): Models object
        progress (ProgressFn, optional): Progress callback function. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame of stage B results with _face_boxesm num_faces_found, and identity labels
    """

    fast_no_identity: bool = getattr(cfg.face, "fast_no_identity", False)
    similarity_treshold: float = cfg.face.sim_tresh
    yolo_batch_size = cfg.face.batch_size

    df_stage_b = df_stage_a.copy()

    if not fast_no_identity and models.facenet is None:
        models.facenet = get_facenet_embedder(ctx)
        log.info(f"[INFO] Facenet embedder (facenet-pytorch) loaded on {ctx.device_str}")

    sharp_with_person = df_stage_b[
        (df_stage_b["focus_label"] == "subject_in_focus") & (df_stage_b["person_count"] >= 1)
    ].copy()
    total = len(sharp_with_person)
    log.info(f"[INFO] Stage B: Processing face detection {total} images")

    if models.yolo_face_model is None and models.retina_detect_fn is None and models.mtcnn is None:
        yolo_enabled, yolo_face_model = build_yolov8_face(
            getattr(cfg.face, "weights_path", "./yolov8n-face-lindevs.pt"), ctx
        )
        if yolo_enabled:
            models.yolo_face_model = yolo_face_model
            log.info(f"[INFO] YOLOv8 face detector loaded on {ctx.device_str}")
        else:
            log.info(
                f"[INFO] YOLOv8 face detector not loaded, falling back to Retina-Face + MTCNN detectors"
            )
            rf_enabled, rf_detect, rf_model = build_retinaface(ctx)
            if rf_enabled:
                models.retina_detect_fn = rf_detect
                models.retina_model = rf_model
            mtcnn_enabled, mtcnn, mtcnn_cpu = build_mtcnn(ctx)
            if mtcnn_enabled:
                models.mtcnn = mtcnn
                models.mtcnn_cpu = mtcnn_cpu

    for k, idx in enumerate(sharp_with_person.index, start=1):
        if progress and (k % 5 == 0):
            progress(k, total, "face-detection")

        row = df_stage_b.loc[idx]
        p = root / row["path"]
        bgr = load_bgr_exif_safe(p)
        if bgr is None:
            continue

        person_count = int(row["person_count"])

        boxes, lms = detect_faces_smart(
            bgr=bgr,
            person_count=person_count,
            rf_enabled=rf_enabled if models.retina_detect_fn is not None else False,
            rf_detect_fn=models.retina_detect_fn,
            rf_model=models.retina_model,
            mtcnn=models.mtcnn,
            mtcnn_cpu=models.mtcnn_cpu,
            yolo_face_model=models.yolo_face_model,
            yolo_batch_size=yolo_batch_size,
            conf=cfg.face.conf,
            max_det=cfg.face.max_det,
            iou=cfg.face.iou,
        )
        boxes_py = to_box_list_strict(boxes)
        df_stage_b.at[idx, "_face_boxes"] = boxes_py
        df_stage_b.at[idx, "num_faces_found"] = int(len(boxes_py))

    if not fast_no_identity:
        single_person_with_faces = df_stage_b[
            (df_stage_b["focus_label"] == "subject_in_focus")
            & (df_stage_b["person_count"] >= 1)
            & (df_stage_b["num_faces_found"] == 1)
        ].copy()

        total_embed = len(single_person_with_faces)
        log.info(f"[INFO] Stage B: Identity clustering on {total_embed} images")

        face_vecs: List[np.ndarray] = []
        img_map: List[Tuple[int, int]] = []

        for j, idx in enumerate(single_person_with_faces.index, start=1):
            if progress and (j % 10 == 0):
                progress(j, total_embed, "face-embedding")
            row = single_person_with_faces.loc[idx]
            p = root / row["path"]
            bgr = load_bgr_exif_safe(p)
            boxes = row["_face_boxes"]

            if not boxes or bgr is None:
                continue

            E = embed_face_batch(models.facenet, bgr, boxes)
            if E is None:
                continue

            for k in range(E.shape[0]):
                face_vecs.append(E[k])
                img_map.append((idx, k))

        if face_vecs:
            E = np.vstack(face_vecs).astype(np.float64, copy=False)
            labels = graph_clusters(E, similarity_treshold=similarity_treshold)
            labels = remap_labels_sequential(labels)

            per_img_labels: Dict[int, List[int]] = {}
            for label, (img_idx, k) in zip(labels, img_map):
                per_img_labels.setdefault(img_idx, []).append(int(label))

            for img_idx, labs in per_img_labels.items():
                df_stage_b.at[img_idx, "_face_id_labels"] = labs
                valid = [l for l in labs if l >= 0]
                df_stage_b.at[img_idx, "identity_group"] = (
                    int(np.bincount(valid).argmax()) if valid else -1
                )

    return df_stage_b
