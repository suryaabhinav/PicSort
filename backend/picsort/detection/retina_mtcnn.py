import importlib
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

from backend.api.logging_config import log
from backend.picsort.config import RuntimeContext
from backend.picsort.detection.yolo_face import detect_faces_yolo_smart
from backend.picsort.utils.helpers import rot90_ccw


def build_retinaface(ctx: RuntimeContext) -> Tuple[bool, Optional[object], Optional[object]]:
    """Build Retina-Face

    Args:
        ctx (RuntimeContext): Runtime context

    Returns:
        Tuple[bool, Optional[object], Optional[object]]: (enabled: bool, detect_fn_or_None, model_or_None)
    """
    try:
        RF_MODULE = importlib.import_module("retinaface.RetinaFace")
        RF_BUILD = getattr(RF_MODULE, "build_model", None)
        RF_DETECT = getattr(RF_MODULE, "detect_faces", None)
        if RF_BUILD is None or RF_DETECT is None:
            raise ImportError("retinaface.RetinaFace missing required functions")
        model = RF_BUILD()
        return True, RF_DETECT, model
    except Exception as e:
        log.warning(f"[WARNING] RetinaFace import/build failed: {e}")
        return False, None, None


def build_mtcnn(ctx: RuntimeContext) -> Tuple[bool, Optional[object], Optional[object]]:
    """Build MTCNN

    Args:
        ctx (RuntimeContext): Runtime context

    Returns:
        Tuple[bool, Optional[object], Optional[object]]: (enabled: bool, detect_fn_or_None, model_or_None)
    """

    try:
        from facenet_pytorch import MTCNN

        mtcnn_device = ctx.device_str
        if mtcnn_device == "mps":
            log.info(
                f"[INFO] MTCNN primary detector forced to CPU due to MPS adaptive pooling limitations"
            )
            mtcnn_device = torch.device("cpu")

        mtcnn = MTCNN(keep_all=True, device=mtcnn_device)

        if mtcnn_device != "cpu":
            mtcnn_cpu = MTCNN(keep_all=True, device=torch.device("cpu"))

        return True, mtcnn, mtcnn_cpu

    except Exception as e:
        log.warning(f"[WARNING] MTCNN initialization failes: {e}")
        return False, None, None


def _rf_parse(dets: dict) -> Tuple[List[Tuple[int, int, int, int]], Optional[np.ndarray]]:
    """Parse RetinaFace results

    Args:
        dets (dict): RetinaFace results

    Returns:
        Tuple[List[Tuple[int, int, int, int]], Optional[np.ndarray]]: (boxes, landmarks)
    """
    boxes, lms_list = [], []
    if not dets or not isinstance(dets, dict):
        return boxes, lms

    for _, d in dets.items():
        fa = d.get("facial_area", None)
        if fa is None or len(fa) < 4:
            continue
        x1, y1, x2, y2 = map(int, fa[:4])
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append((x1, y1, x2, y2))
        lm = d.get("landmarks", {})
        order = ["left_eye", "right_eye", "nose", "mouth_left", "mouth_right"]
        pts = []
        for k in order:
            v = lm.get(k, None)
            if v is None or len(v) < 2:
                continue
            pts.append((float(v[0]), float(v[1])))
        if len(pts) == 5:
            lms_list.append(np.array(pts, dtype=np.float32))
    lms = np.stack(lms_list, 0) if lms_list else None
    return boxes, lms


def detect_faces_smart(
    bgr: np.ndarray,
    person_count: int,
    rf_enabled: bool,
    rf_detect_fn: Optional[object],
    rf_model: Optional[object],
    mtcnn: Optional[object],
    mtcnn_cpu: Optional[object],
    yolo_face_model: Optional[object],
    yolo_batch_size: int,
    conf: float,
    max_det: int,
    iou: float,
) -> Tuple[List[Tuple[int, int, int, int]], Optional[np.ndarray]]:
    """Smart face detection with rotation only when needed

    Args:
        bgr (np.ndarray): Input image
        person_count (int): Number of people in the image
        rf_enabled (bool): Whether RetinaFace is enabled
        rf_detect_fn (Optional[object]): RetinaFace detection function
        rf_model (Optional[object]): RetinaFace model
        mtcnn (Optional[object]): MTCNN model
        mtcnn_cpu (Optional[object]): MTCNN model on CPU
        yolo_face_model (Optional[object]): YOLOv8 face model
        yolo_batch_size (int): YOLOv8 batch size
        conf (float): YOLOv8 confidence threshold
        max_det (int): YOLOv8 maximum number of detections
        iou (float): YOLOv8 IOU threshold

    Returns:
        Tuple[List[Tuple[int, int, int, int]], Optional[np.ndarray]]: (boxes, landmarks)
    """

    H, W = bgr.shape[:2]

    if yolo_face_model is not None and person_count > 0:
        boxes, lms = detect_faces_yolo_smart(
            bgr=bgr,
            person_count=person_count,
            yolo_face_model=yolo_face_model,
            yolo_batch_size=yolo_batch_size,
            conf=conf,
            max_det=max_det,
            iou=iou,
        )
        return boxes, lms

    if rf_enabled and rf_detect_fn is not None and rf_model is not None:
        try:
            detectors = rf_detect_fn(bgr, model=rf_model, threshold=conf, max_dets=max_det)
            boxes, lms = _rf_parse(detectors)
            if boxes:
                return boxes, lms

            if person_count >= 1:
                bgr90 = rot90_ccw(bgr)
                detectors90 = rf_detect_fn(bgr90, model=rf_model, threshold=conf, max_dets=max_det)
                boxes90, lms90 = _rf_parse(detectors90)
                mapped_boxes = []
                mapped_lms = None
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
                            mapped_lms_list.append(np.array(pts), dtype=np.float32)

                        mapped_lms = np.stack(mapped_lms_list, axis=0)
                    return mapped_boxes, mapped_lms
        except Exception as e:
            log.warning(f"[WARNING] RetinaFace smart detection failed: {e}")

    if mtcnn is not None and person_count >= 1:
        try:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            detectors = []
            if mtcnn is not None:
                detectors.append(mtcnn)
            if mtcnn_cpu is not None and mtcnn_cpu is not mtcnn:
                detectors.append(mtcnn_cpu)

            for det in detectors:
                try:
                    boxes_arr, probs, landmarks = det.detect(rgb, landmarks=True)
                    if boxes_arr is None or len(boxes_arr) == 0:
                        continue
                    keep = [i for i, p in enumerate(probs) if (p is not None and p >= conf)]
                    if len(keep) == 0:
                        continue
                    bx = boxes_arr[keep].astype(int).tolist()
                    lm = landmarks[keep] if landmarks is not None else None
                    lms = lm.astype(np.float32) if lm is not None else None
                    bx_tuple = [(int(b[0]), int(b[1]), int(b[2]), int(b[3])) for b in bx]
                    return bx_tuple, lms
                except RuntimeError as e:
                    msg = str(e)
                    dev = getattr(getattr(det, "devive", None), "type", None)
                    if "Adaptive pool MPS" in msg and mtcnn_cpu is not None and dev == "mps":
                        log.info(f"[INFO] MTCNN on MPS hit adaptive pool bug, retrying on CPU")
                        continue
                    log.warning("[WARNING] MTCNN fallback failed: " + msg)
                except Exception as e:
                    log.warning(f"[WARNING] fallback failed: {e}")
        except Exception as e:
            log.warning(f"[WARNING] MTCNN fallback failed: {e}")

    return [], None
