import logging
from typing import List, Optional, Tuple

import numpy as np
from ultralytics.models.yolo.model import YOLO

from api.logging_config import get_logger, log
from picsort.config import RuntimeContext
from picsort.utils.helpers import rot90_ccw


def build_yolov8_face(weights_path: str, ctx: RuntimeContext) -> Tuple[bool, object]:
    """Build YOLOv8 face detector

    Args:
        weights_path (str): Path to the weights file

    Returns:
        Tuple[bool, object]: (enabled: bool, model_or_None)
    """

    try:
        model = YOLO(weights_path)
        model.to(ctx.device_str)
        try:
            model.fuse()
        except Exception:
            pass
        return True, model
    except Exception as e:
        log.warning(f"YOLOv8-Face build failed: {e}")
        return False, None


def detect_faces_yolo(
    bgr: np.ndarray,
    yolo_face_model: object,
    yolo_batch_size: int,
    conf: float,
    max_det: int,
    iou: float,
) -> Tuple[List[Tuple[int, int, int, int]], Optional[np.ndarray]]:
    """YOLOv8-Face detection

    Args:
        bgr (np.ndarray): Input image
        yolo_face_model (object): YOLOv8 face detector model
        yolo_batch_size (int): YOLOv8 batch size
        conf (float): YOLOv8 confidence threshold
        max_det (int): YOLOv8 maximum number of detections
        iou (float): YOLOv8 IOU threshold

    Returns:
        Tuple[List[Tuple[int, int, int, int]], Optional[np.ndarray]]: (boxes, landmarks)
    """

    if yolo_face_model is None:
        return [], None

    results = yolo_face_model(
        bgr,
        conf=conf,
        batch=yolo_batch_size,
        max_det=max_det,
        iou=iou,
        verbose=False,
    )[0]

    boxes_out: List[Tuple[int, int, int, int]] = []

    if results.boxes is None or len(results.boxes) == 0:
        return boxes_out, None

    xyxy = results.boxes.xyxy.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()

    for (x1, y1, x2, y2), c in zip(xyxy, confs):
        if c < conf:
            continue
        x1i = int(max(0, np.floor(x1)))
        y1i = int(max(0, np.floor(y1)))
        x2i = int(min(bgr.shape[1], np.ceil(x2)))
        y2i = int(min(bgr.shape[0], np.ceil(y2)))
        if x2i <= x1i or y2i <= y1i:
            continue
        boxes_out.append((x1i, y1i, x2i, y2i))

    return boxes_out, None


def detect_faces_yolo_smart(
    bgr: np.ndarray,
    person_count: int,
    yolo_face_model: object,
    yolo_batch_size: int,
    conf: float,
    max_det: int,
    iou: float,
) -> Tuple[List[Tuple[int, int, int, int]], Optional[np.ndarray]]:
    """YOLOv8-Face detection with optional 90 degree CCW retry when person_count >= 1

    Args:
        bgr (np.ndarray): Input image
        person_count (int): Number of people in the image
        yolo_face_model (object): YOLOv8 face detector model
        yolo_batch_size (int): YOLOv8 batch size
        conf (float): YOLOv8 confidence threshold
        max_det (float): YOLOv8 maximum number of detections
        iou (float): YOLOv8 IOU threshold

    Returns:
        Tuple[List[Tuple[int, int, int, int]], Optional[np.ndarray]]: (boxes, landmarks)
    """

    H, W = bgr.shape[:2]

    if person_count >= 1:
        boxes, lms = detect_faces_yolo(
            bgr=bgr,
            yolo_face_model=yolo_face_model,
            yolo_batch_size=yolo_batch_size,
            conf=conf,
            max_det=max_det,
            iou=iou,
        )

        if boxes:
            return boxes, lms

        bgr90 = rot90_ccw(bgr)
        boxes90, lms90 = detect_faces_yolo(
            bgr=bgr90,
            yolo_face_model=yolo_face_model,
            yolo_batch_size=yolo_batch_size,
            conf=conf,
            max_det=max_det,
            iou=iou,
        )

        mapped_boxes: List[Tuple[int, int, int, int]] = []
        mapped_lms: Optional[np.ndarray] = None

        if boxes90:
            for x1, y1, x2, y2 in boxes90:
                X1, Y1 = y1, W - x2
                X2, Y2 = y2, W - x1
                mapped_boxes.append((int(X1), int(Y1), int(X2), int(Y2)))

            if lms90 is not None:
                mapped_lms_list = []
                for lm in lms90:
                    pts = []
                    for x, y in lm:
                        X, Y = y, W - x
                        pts.append((float(X), float(Y)))
                    mapped_lms_list.append(np.array(pts, dtype=np.float32))

                mapped_lms = np.stack(mapped_lms_list, axis=0)

        return mapped_boxes, mapped_lms

    return [], None
