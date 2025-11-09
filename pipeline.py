#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse, hashlib, sys, logging, importlib
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Mapping, Union
import shutil

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm
import cv2
import torch
from sklearn.metrics.pairwise import cosine_similarity
import hdbscan


# ========================= Logging =========================


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

# ========================= Device =========================


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        try:
            _ = torch.zeros(1, device="mps")
            return torch.device("mps")
        except Exception:
            pass
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = select_device()
DEV_STR = DEVICE.type
print(f"[INFO] Hardware Setup Complete. Using device: {DEV_STR}")
if DEV_STR == "mps":
    print("[INFO] Apple MPS detected: batch processing optimized for M1/M2/M3")

TORCH_INFERENCE = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".gif"}

# ========================= IO Utils =========================


def list_images(root: Path) -> List[Path]:
    return [
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in VALID_EXTS and not p.name.startswith("._")
    ]


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for ch in iter(lambda: f.read(8192), b""):
            h.update(ch)
    return h.hexdigest()


def load_pil_exif_rgb(path: Path) -> Image.Image:
    im = Image.open(path)
    im = ImageOps.exif_transpose(im).convert("RGB")
    return im


def load_bgr_exif_safe(path: Path) -> Optional[np.ndarray]:
    try:
        pil = load_pil_exif_rgb(path)
        arr = np.array(pil)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except FileNotFoundError:
        log.warning("Image missing on disk: %s", path)
        return None
    except Exception:
        try:
            arr = np.fromfile(str(path), dtype=np.uint8)
        except FileNotFoundError:
            log.warning("Image missing on disk: %s", path)
            return None
        if arr.size == 0:
            return None
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            log.warning("OpenCV decode failed on %s", path)
        return bgr


# ========================= Focus Metrics =========================


def tenengrad(gray: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    g2 = gx * gx + gy * gy
    if mask is not None:
        g2 = g2[mask]
    return float(np.mean(g2)) if g2.size else 0.0


def contrast(gray: np.ndarray, mask: Optional[np.ndarray] = None, eps=1e-6) -> float:
    roi = gray[mask] if mask is not None else gray.reshape(-1)
    return float(np.std(roi) + eps)


def multiscale_focus(gray: np.ndarray, mask: Optional[np.ndarray] = None, levels=3) -> float:
    scores = []
    g = gray
    m = mask
    for _ in range(levels):
        scores.append(tenengrad(g, m))
        if min(g.shape[:2]) < 64:
            break
        g = cv2.pyrDown(g)
        if m is not None:
            m = cv2.resize(
                m.astype(np.uint8), (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        if m is not None and m.sum() == 0:
            break
    return float(np.mean(scores))


def masked_focus(gray: np.ndarray, mask: Optional[np.ndarray]) -> float:
    f = multiscale_focus(gray, mask, 3)
    c = contrast(gray, mask)
    return f / c if c > 0 else 0.0


# ========================= YOLO11 (Ultralytics) =========================

_HAS_YOLO = False
try:
    from ultralytics import YOLO

    _HAS_YOLO = True
except Exception:
    log.warning("ultralytics not available; subject/person detection disabled")


class YOLOProcessor:
    """Unified YOLO processor for segmentation and person detection with batch support"""

    def __init__(
        self,
        seg_model="yolo11m-seg.pt",
        det_model="yolo11m.pt",
        person_conf=0.50,
        person_min_area=0.01,
    ):
        self.seg_ok = False
        self.det_ok = False
        self.seg_model = None
        self.det_model = None
        self.person_conf = person_conf
        self.person_min_area = person_min_area

        if _HAS_YOLO:
            try:
                self.seg_model = YOLO(seg_model)
                if DEV_STR in ("mps", "cuda"):
                    self.seg_model.to(DEV_STR)
                try:
                    self.seg_model.fuse()
                except Exception:
                    pass
                self.seg_ok = True
                print(f"[INFO] YOLO11-seg initialized: {seg_model}")
            except Exception as e:
                log.warning("YOLO11-seg init failed: %s", e)

            try:
                self.det_model = YOLO(det_model)
                if DEV_STR in ("mps", "cuda"):
                    self.det_model.to(DEV_STR)
                try:
                    self.det_model.fuse()
                except Exception:
                    pass
                self.det_ok = True
                print(f"[INFO] YOLO11-detect initialized: {det_model} (conf={person_conf:.2f})")
            except Exception as e:
                log.warning("YOLO11-detect init failed: %s", e)

    def process_batch(self, image_paths: List[Path], batch_size: int = 32) -> Dict[str, Dict]:
        """Process images in batches for masks and person counts"""
        results = {}

        if not self.seg_ok and not self.det_ok:
            # Fallback to ellipse masks
            for p in image_paths:
                try:
                    bgr = load_bgr_exif_safe(p)
                    if bgr is not None:
                        results[p.name] = {
                            "mask": self._ellipse(bgr),
                            "person_count": 0,
                            "mask_type": "ellipse",
                        }
                except Exception:
                    pass
            return results

        # Batch processing with YOLO
        total_batches = (len(image_paths) + batch_size - 1) // batch_size

        with tqdm(total=total_batches, desc="Stage A: YOLO batch", unit="batch") as pbar:
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i : i + batch_size]

                # If segmentation model is available, use it for BOTH mask and person_count
                if self.seg_ok:
                    try:
                        seg_results = self.seg_model.predict(
                            source=[str(p) for p in batch_paths],
                            batch=batch_size,
                            imgsz=640,
                            conf=0.4,
                            device=DEV_STR,  # "mps", "cuda" or "cpu"
                            verbose=False,
                            stream=False,
                        )

                        for path, result in zip(batch_paths, seg_results):
                            # 1) Subject mask (your existing logic)
                            mask = self._extract_mask(result)

                            # 2) Person count from segmentation boxes
                            person_count = 0
                            boxes = result.boxes
                            person_conf_thresh = 0.7
                            if boxes is not None and len(boxes) > 0:
                                cls = boxes.cls.cpu().numpy().astype(int)  # (N,)
                                conf = boxes.conf.cpu().numpy()  # (N,)
                                # print("cls:", cls)
                                # print("conf:", conf)

                                # Map class index -> name
                                names = result.names  # {0: 'person', 1: 'bicycle', ...}

                                # Find the index of the "person" class
                                person_ids = [k for k, v in names.items() if v == "person"]
                                if person_ids:
                                    pid = person_ids[0]

                                    # Boolean mask: is person AND confident enough
                                    is_person = cls == pid
                                    is_confident = conf >= person_conf_thresh
                                    keep = is_person & is_confident

                                    person_count = int(keep.sum())

                                    # Optional debug:
                                    # print("pid:", pid)
                                    # print("cls:", cls)
                                    # print("conf:", conf)
                                    # print("keep:", keep)

                            # 3) Fallback mask if seg failed
                            if mask is None:
                                bgr = load_bgr_exif_safe(path)
                                if bgr is not None:
                                    mask = self._ellipse(bgr)
                                    mask_type = "ellipse"
                                else:
                                    mask_type = "none"
                            else:
                                mask_type = "yolo11-seg"

                            results[path.name] = {
                                "mask": mask,
                                "person_count": int(person_count),
                                "mask_type": mask_type,
                            }

                    except Exception as e:
                        log.warning(f"YOLO seg batch failed: {e}")
                        # Hard fallback: ellipse masks + 0 persons for this whole batch
                        for path in batch_paths:
                            bgr = load_bgr_exif_safe(path)
                            if bgr is not None:
                                results[path.name] = {
                                    "mask": self._ellipse(bgr),
                                    "person_count": 0,
                                    "mask_type": "ellipse",
                                }
                else:
                    # No seg model at all: pure ellipse fallback
                    for path in batch_paths:
                        bgr = load_bgr_exif_safe(path)
                        if bgr is not None:
                            results[path.name] = {
                                "mask": self._ellipse(bgr),
                                "person_count": 0,
                                "mask_type": "ellipse",
                            }

                pbar.update(1)

        return results

    def _extract_mask(self, result) -> Optional[np.ndarray]:
        """Extract segmentation mask from YOLO result"""
        try:
            if result.masks is None or len(result.masks.data) == 0:
                return None

            masks = result.masks.data
            if hasattr(masks, "device") and masks.device.type != "cpu":
                masks = masks.cpu()
            masks = masks.numpy().astype(bool)

            # Get original image size
            orig_shape = result.orig_shape
            h, w = orig_shape[0], orig_shape[1]

            # Resize masks if needed
            if masks.shape[1] != h or masks.shape[2] != w:
                resized = []
                for m in masks:
                    m8 = m.astype(np.uint8) * 255
                    m8 = cv2.resize(m8, (w, h), interpolation=cv2.INTER_NEAREST)
                    resized.append(m8.astype(bool))
                masks = np.stack(resized, 0)

            # Combine all masks
            combined = np.logical_or.reduce(masks)
            return combined
        except Exception as e:
            log.warning(f"Mask extraction failed: {e}")
            return None

    def _count_persons(self, result, path: Path) -> int:
        """Count persons from YOLO detection result"""
        try:
            if result.boxes is None:
                return 0

            # Get image dimensions for area calculation
            bgr = load_bgr_exif_safe(path)
            if bgr is None:
                return 0

            H, W = bgr.shape[:2]
            area_min = H * W * self.person_min_area

            count = 0
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)
                if (x2 - x1) * (y2 - y1) >= area_min:
                    count += 1

            return count
        except Exception:
            return 0

    @staticmethod
    def _ellipse(bgr: np.ndarray) -> np.ndarray:
        """Fallback ellipse mask"""
        H, W = bgr.shape[:2]
        yy, xx = np.mgrid[0:H, 0:W]
        cx, cy = W / 2.0, H / 2.0
        rx, ry = W * 0.35, H * 0.35
        return ((xx - cx) ** 2) / (rx**2) + ((yy - cy) ** 2) / (ry**2) <= 1.0


# ========================= RetinaFace + MTCNN =========================

# ================ YOLO =====================


def build_yolov8_face(weights_path: str):
    """
    Build YOLOv8-Face detector.
    Returns: (enabled: bool, model_or_None)
    """
    try:
        model = YOLO(weights_path)
        model.to(DEVICE)
        # Optional: fuse for slightly faster inference
        try:
            model.fuse()
        except Exception:
            pass
        log.info("YOLOv8-Face loaded on device=%s", DEVICE)
        return True, model
    except Exception as e:
        log.warning("YOLOv8-Face build failed: %s", e)
        return False, None


def detect_faces_yolo(
    bgr: np.ndarray,
    yolo_face_model,
    conf_thresh: float = 0.5,
    max_det: int = 300,
    yolo_batch_size: int = 32,
) -> Tuple[List[Tuple[int, int, int, int]], Optional[np.ndarray]]:
    """
    Run YOLOv8-Face on a single BGR image.
    Returns (boxes, lms) where:
      boxes: List[(x1,y1,x2,y2)]
      lms:   None for now (no landmarks in lindevs models)
    """
    if yolo_face_model is None:
        return [], None

    # Ultralytics YOLO API: model(image, conf=..., max_det=...) -> list[Results]
    results = yolo_face_model(
        bgr,
        conf=conf_thresh,
        batch=yolo_batch_size,
        max_det=max_det,
        verbose=False,
    )[0]

    boxes_out: List[Tuple[int, int, int, int]] = []

    if results.boxes is None or len(results.boxes) == 0:
        return boxes_out, None

    # results.boxes.xyxy: (N,4), results.boxes.conf: (N,)
    xyxy = results.boxes.xyxy.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()

    for (x1, y1, x2, y2), c in zip(xyxy, confs):
        if c < conf_thresh:
            continue
        x1i = int(max(0, np.floor(x1)))
        y1i = int(max(0, np.floor(y1)))
        x2i = int(min(bgr.shape[1], np.ceil(x2)))
        y2i = int(min(bgr.shape[0], np.ceil(y2)))
        if x2i <= x1i or y2i <= y1i:
            continue
        boxes_out.append((x1i, y1i, x2i, y2i))

    return boxes_out, None  # landmarks=None for now


def detect_faces_yolo_smart(
    bgr: np.ndarray,
    person_count: int,
    yolo_face_model,
    conf_thresh: float = 0.5,
    yolo_batch_size: int = 32,
) -> Tuple[List[Tuple[int, int, int, int]], Optional[np.ndarray]]:
    """
    YOLOv8-Face detection with optional 90° CCW retry when person_count>=1.
    """
    H, W = bgr.shape[:2]

    # 1) Normal orientation
    boxes, lms = detect_faces_yolo(
        bgr, yolo_face_model, conf_thresh=conf_thresh, yolo_batch_size=yolo_batch_size
    )
    if boxes:
        return boxes, lms

    # 2) Smart rotation: only if we *expect* faces
    if person_count >= 1:
        bgr90 = _rot90_ccw(bgr)
        boxes90, lms90 = detect_faces_yolo(bgr90, yolo_face_model, conf_thresh=conf_thresh)

        mapped_boxes: List[Tuple[int, int, int, int]] = []
        mapped_lms: Optional[np.ndarray] = None

        if boxes90:
            for x1, y1, x2, y2 in boxes90:
                # Your original mapping
                X1, Y1 = y1, W - x2
                X2, Y2 = y2, W - x1
                mapped_boxes.append((int(X1), int(Y1), int(X2), int(Y2)))

            # If we ever get landmarks in YOLO, we can map similarly:
            if lms90 is not None:
                mapped_lms_list = []
                for lm in lms90:
                    pts = []
                    for x, y in lm:
                        X, Y = y, W - x
                        pts.append([float(X), float(Y)])
                    mapped_lms_list.append(np.array(pts, dtype=np.float32))
                mapped_lms = np.stack(mapped_lms_list, 0)

        return mapped_boxes, mapped_lms

    # No faces found, even after rotation
    return [], None


# ==========================================================


def build_retinaface() -> Tuple[bool, Optional[object], Optional[object]]:
    """Returns (use_retina, RF_DETECT, RF_MODEL)"""
    try:
        RF_MODULE = importlib.import_module("retinaface.RetinaFace")
        RF_BUILD = getattr(RF_MODULE, "build_model", None)
        RF_DETECT = getattr(RF_MODULE, "detect_faces", None)
        if RF_BUILD is None or RF_DETECT is None:
            raise ImportError("retinaface.RetinaFace missing required functions")
        model = RF_BUILD()
        return True, RF_DETECT, model
    except Exception as e:
        log.warning("RetinaFace import/build failed: %s", e)
        return False, None, None


def _rf_parse(dets: dict) -> Tuple[List[Tuple[int, int, int, int]], Optional[np.ndarray]]:
    boxes, lms_list = [], []
    if not dets or not isinstance(dets, dict):
        return boxes, None
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
            if v is None:
                pts = []
                break
            pts.append([float(v[0]), float(v[1])])
        if len(pts) == 5:
            lms_list.append(np.array(pts, dtype=np.float32))
    lms = np.stack(lms_list, 0) if lms_list else None
    return boxes, lms


def _rot90_ccw(bgr: np.ndarray) -> np.ndarray:
    """Rotate 90 degrees counter-clockwise"""
    return np.ascontiguousarray(np.rot90(bgr, k=1))


def detect_faces_smart(
    bgr: np.ndarray,
    person_count: int,
    rf_enabled: bool,
    rf_detect_fn,
    rf_model,
    mtcnn,
    mtcnn_cpu=None,
    yolo_face_model=None,
    yolo_batch_size: int = 32,
) -> Tuple[List[Tuple[int, int, int, int]], Optional[np.ndarray]]:
    """Smart face detection with rotation only when needed"""
    H, W = bgr.shape[:2]

    # 0) YOLOv8-Face preferred if available
    if yolo_face_model is not None:
        boxes, lms = detect_faces_yolo_smart(
            bgr, person_count, yolo_face_model, conf_thresh=0.5, yolo_batch_size=32
        )
        # if boxes:
        return boxes, lms

    # 1) RetinaFace - normal orientation
    if rf_enabled and rf_detect_fn is not None and rf_model is not None:
        try:
            dets = rf_detect_fn(bgr, threshold=0.6, model=rf_model)
            boxes, lms = _rf_parse(dets)
            if boxes:
                return boxes, lms

            # Smart rotation: only if person detected but no faces
            if person_count >= 1:
                # Try 90° CCW (most common for portraits)
                bgr90 = _rot90_ccw(bgr)
                dets90 = rf_detect_fn(bgr90, threshold=0.6, model=rf_model)
                boxes90, lms90 = _rf_parse(dets90)
                mapped_boxes = []
                mapped_lms = None
                if boxes90:
                    # Remap coordinates back to original orientation
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
                                pts.append([float(X), float(Y)])
                            mapped_lms_list.append(np.array(pts, dtype=np.float32))
                        mapped_lms = np.stack(mapped_lms_list, 0)
                return mapped_boxes, mapped_lms
        except Exception as e:
            log.warning("RetinaFace detection failed: %s", e)

    # 2) MTCNN fallback
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
            keep = [i for i, p in enumerate(probs) if (p is not None and p >= 0.80)]
            if not keep:
                continue
            bx = boxes_arr[keep].astype(int).tolist()
            lm = landmarks[keep] if landmarks is not None else None
            lms = lm.astype(np.float32) if lm is not None else None
            bx_tuple = [(int(b[0]), int(b[1]), int(b[2]), int(b[3])) for b in bx]
            return bx_tuple, lms
        except RuntimeError as e:
            msg = str(e)
            dev = getattr(getattr(det, "device", None), "type", None)
            if "Adaptive pool MPS" in msg and mtcnn_cpu is not None and dev == "mps":
                log.info("MTCNN on MPS hit adaptive pool bug; retrying on CPU.")
                continue
            log.warning("MTCNN fallback failed: %s", e)
        except Exception as e:
            log.warning("MTCNN fallback failed: %s", e)

    return [], None


# ========================= Helpers =========================


def to_box_list_strict(boxes) -> List[Tuple[int, int, int, int]]:
    """Return a pure-Python list of (x1,y1,x2,y2)"""
    if boxes is None:
        return []
    try:
        arr = np.asarray(boxes)
        if arr.size == 0:
            return []
        arr = arr.reshape(-1, 4).astype(int)
        return [
            (int(x1), int(y1), int(x2), int(y2))
            for x1, y1, x2, y2 in arr.tolist()
            if x2 > x1 and y2 > y1
        ]
    except Exception:
        out = []
        for b in boxes:
            if isinstance(b, (list, tuple)) and len(b) >= 4:
                x1, y1, x2, y2 = map(int, b[:4])
                if x2 > x1 and y2 > y1:
                    out.append((x1, y1, x2, y2))
        return out


# ========================= Face Embeddings =========================


def get_facenet_embedder(device: torch.device):
    from facenet_pytorch import InceptionResnetV1

    model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    return model


def embed_faces_batch(
    embedder, bgr: np.ndarray, boxes: List[Tuple[int, int, int, int]]
) -> Optional[np.ndarray]:
    """Embed multiple faces from single image"""
    if not boxes:
        return None
    import torchvision.transforms as T

    trans = T.Compose(
        [
            T.ToTensor(),
            T.Resize((160, 160), antialias=True),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    rgbs = []
    for x1, y1, x2, y2 in boxes:
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(bgr.shape[1], x2)
        y2 = min(bgr.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            continue
        if (x2 - x1) < 40 or (y2 - y1) < 40:
            continue
        crop = bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        rgbs.append(trans(Image.fromarray(rgb)))
    if not rgbs:
        return None

    batch = torch.stack(rgbs).to(DEVICE, memory_format=torch.channels_last)
    with TORCH_INFERENCE():
        emb = embedder(batch).detach().cpu().numpy().astype(np.float32)
    # L2-normalize
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12

    return emb


def cluster_hdbscan(E: np.ndarray, min_cluster_size: int = 2) -> np.ndarray:
    """Cluster L2-normalized embeddings with HDBSCAN on Euclidean distance."""
    n = E.shape[0]
    if n < min_cluster_size:
        return np.full((n,), -1, dtype=int)

    # Ensure float64 for HDBSCAN's internal calculations
    E64 = E.astype(np.float64, copy=False)

    cl = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        # Let min_samples default to min_cluster_size; more conservative, less fragmentation
        metric="euclidean",
        cluster_selection_method="eom",  # more "merged" clusters vs leaf
        core_dist_n_jobs=1,
    )
    labels = cl.fit_predict(E64)
    return labels


def graph_clusters(E: np.ndarray, sim_thresh: float = 0.65) -> np.ndarray:
    n = E.shape[0]
    if n == 0:
        return np.array([], dtype=int)
    E64 = E.astype(np.float64, copy=False)
    sim = cosine_similarity(E64)
    np.fill_diagonal(sim, 1.0)
    adj = (sim >= sim_thresh).astype(np.uint8)
    labels = -np.ones(n, dtype=int)
    cid = 0
    for i in range(n):
        if labels[i] != -1:
            continue
        stack = [i]
        labels[i] = cid
        while stack:
            u = stack.pop()
            neigh = np.where(adj[u] == 1)[0]
            for v in neigh:
                if labels[v] == -1:
                    labels[v] = cid
                    stack.append(v)
        cid += 1
    return labels


def graph_clusters_v2(
    E: np.ndarray,
    sim_thresh: float = 0.65,
    k: int = 10,
    min_cluster_size: int = 1,
) -> np.ndarray:
    n = E.shape[0]
    if n == 0:
        return np.array([], dtype=int)
    E64 = E.astype(np.float64, copy=False)
    sim = cosine_similarity(E64)
    np.fill_diagonal(sim, 1.0)

    if k >= n:
        k = n - 1 if n > 1 else 0

    if k > 0:
        nn_idx = np.argsort(sim, axis=1)[:, -(k + 1) :]
    else:
        nn_idx = None

    adj = np.zeros((n, n), dtype=bool)

    if k > 0:
        for i in range(n):
            neighbors = nn_idx[i]
            neighbors = neighbors[neighbors != i]  # drop self
            for j in neighbors:
                if sim[i, j] >= sim_thresh:
                    adj[i, j] = True

        adj_mutual = adj & adj.T
        adj = adj_mutual
    else:
        adj = sim >= sim_thresh

    labels = -np.ones(n, dtype=int)
    cid = 0
    for i in range(n):
        if labels[i] != -1:
            continue
        stack = [i]
        labels[i] = cid
        while stack:
            u = stack.pop()
            neigh = np.where(adj[u])[0]
            for v in neigh:
                if labels[v] == -1:
                    labels[v] = cid
                    stack.append(v)
        cid += 1

    if min_cluster_size > 1:
        unique, counts = np.unique(labels, return_counts=True)
        small = {cl for cl, c in zip(unique, counts) if (cl != -1 and c < min_cluster_size)}
        if small:
            for i, lbl in enumerate(labels):
                if lbl in small:
                    labels[i] = -1

            unique = sorted({lbl for lbl in labels if lbl != -1})
            remap = {old: new for new, old in enumerate(unique)}
            for i, lbl in enumerate(labels):
                if lbl != -1:
                    labels[i] = remap[lbl]

    return labels


def remap_labels_sequential(labels: np.ndarray) -> np.ndarray:
    mapping = {}
    nxt = 0
    out = labels.copy()
    for i, lab in enumerate(labels):
        if lab >= 0:
            if lab not in mapping:
                mapping[lab] = nxt
                nxt += 1
            out[i] = mapping[lab]
    return out


# ========================= OpenCLIP (scenes) =========================


def get_openclip(device: torch.device):
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="laion2b_s34b_b88k", device=device
    )
    print("[INFO] OpenCLIP ViT-B/16 (LAION2B) ready on", device.type)
    return model, preprocess


def embed_images_openclip_batch(
    model, preprocess, device, paths: List[Path], batch_size: int = 32
) -> Dict[str, np.ndarray]:
    """Batch embed images with OpenCLIP"""
    embs = {}
    total_batches = (len(paths) + batch_size - 1) // batch_size

    with tqdm(total=total_batches, desc="Stage C: scene-emb", unit="batch") as pbar:
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i : i + batch_size]
            batch_tensors = []
            valid_paths = []

            for p in batch_paths:
                try:
                    pil = load_pil_exif_rgb(p)
                    im = preprocess(pil).unsqueeze(0)
                    batch_tensors.append(im)
                    valid_paths.append(p)
                except Exception as e:
                    log.warning("OpenCLIP preprocess failed on %s: %s", p.name, e)

            if batch_tensors:
                try:
                    batch = torch.cat(batch_tensors, dim=0).to(device)
                    with TORCH_INFERENCE():
                        feat = model.encode_image(batch)
                    feat_np = feat.detach().cpu().numpy().astype(np.float32)

                    for p, v in zip(valid_paths, feat_np):
                        embs[p.name] = v.reshape(-1)
                except Exception as e:
                    log.warning("OpenCLIP batch encode failed: %s", e)

            pbar.update(1)

    return embs


# ========================= MD5 Deduplication =========================


def calculate_md5_batch(image_paths: List[Path]) -> pd.DataFrame:
    """Calculate MD5 hashes for all images"""
    print("[INFO] Calculating MD5 hashes for deduplication...")
    rows = []
    for p in tqdm(image_paths, desc="MD5 calculation", unit="img"):
        try:
            md5 = md5_file(p)
            rows.append({"path": p.name, "md5": md5})
        except Exception as e:
            log.warning("MD5 failed for %s: %s", p.name, e)

    df = pd.DataFrame(rows)

    # Find duplicates
    df["is_duplicate"] = df.duplicated(subset=["md5"], keep="first")
    n_dupes = df["is_duplicate"].sum()

    if n_dupes > 0:
        print(f"[INFO] Found {n_dupes} duplicate images (by MD5)")
        print(f"[INFO] Will process {len(df) - n_dupes} unique images")

    return df


def broadcast_to_duplicates(df_processed: pd.DataFrame, df_md5: pd.DataFrame) -> pd.DataFrame:
    """Broadcast results to duplicate images based on md5 matching"""

    print(f"[INFO] Broadcasting results to {df_md5['is_duplicate'].sum()} duplicate images...")

    md5_to_metrics = {}

    for _, row in df_processed.iterrows():
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

    for _, row in df_md5.iterrows():
        md5_hash = row["md5"]
        path = row["path"]

        result_row = {"path": path, "md5": md5_hash}

        if md5_hash in md5_to_metrics:
            result_row.update(md5_to_metrics[md5_hash])
        else:
            log.warning(f"No metrics found for md5 {md5_hash} (path: {path})")
            result_row.update(
                {
                    "subject_sharpness": 0.0,
                    "background_sharpness": 0.0,
                    "num_faces_found": 0,
                    "person_count": 0,
                    "focus_label": "unknown",
                    "mask_used": "none",
                    "scene_group": -100,
                    "identity_group": -10,
                    "_face_boxes": [],
                    "_face_id_labels": [],
                    "final_group_name": "30_Outliers/02_Unclassified_Sharp_NoFace",
                }
            )

        result_rows.append(result_row)

    df_final = pd.DataFrame(result_rows)

    print(f"[INFO] Final dataframe has {len(df_final)} total images")
    return df_final


# ========================= Stage A =========================


def run_stage_a(
    root: Path,
    imgs: List[Path],
    df_md5: pd.DataFrame,
    t_subj=20.0,
    t_bg=90.0,
    closeness=25.0,
    person_conf=0.50,
    person_min_area=0.01,
    yolo_batch_size=32,
) -> pd.DataFrame:

    # Filter to unique images only
    unique_paths_set = set(df_md5[~df_md5["is_duplicate"]]["path"].tolist())
    imgs_to_process = [p for p in imgs if p.name in unique_paths_set]

    print(
        f"[INFO] Stage A: Processing {len(imgs_to_process)} unique images (skipping {len(imgs) - len(imgs_to_process)} duplicates)"
    )

    # Initialize YOLO processor
    yolo = YOLOProcessor(person_conf=person_conf, person_min_area=person_min_area)

    # Batch process with YOLO
    yolo_results = yolo.process_batch(imgs_to_process, batch_size=yolo_batch_size)

    # Calculate focus metrics
    rows = []
    print("[INFO] Calculating focus metrics...")
    for p in tqdm(imgs_to_process, desc="Stage A: focus metrics", unit="img"):
        bgr = load_bgr_exif_safe(p)
        if bgr is None:
            continue

        path_key = p.name
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        yolo_data = yolo_results.get(path_key, None)

        if yolo_data is None or yolo_data.get("mask") is None:
            # Fallback to ellipse
            mask = YOLOProcessor._ellipse(bgr)
            yolo_data = {"mask": mask, "person_count": 0, "mask_type": "ellipse"}
        else:
            mask = yolo_data["mask"]

        bg_mask = ~mask

        subj_s = masked_focus(gray, mask)
        bg_s = masked_focus(gray, bg_mask)

        # Classify focus
        label = "subject_in_focus"
        if subj_s < t_subj:
            if bg_s < (t_subj / 2):
                label = "overall_soft_low_quality"
            else:
                label = "subject_soft_general"
        elif bg_s > t_bg and abs(subj_s - bg_s) < closeness:
            label = "overall_sharp_flat_focus"

        rows.append(
            {
                "path": path_key,
                "subject_sharpness": round(subj_s, 2),
                "background_sharpness": round(bg_s, 2),
                "eye_sharpness": 0.0,  # COMMENTED OUT - unused
                "num_eyes_found": 0,  # COMMENTED OUT - unused
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

    df = pd.DataFrame(rows)

    # Add MD5 from df_md5
    df = df.merge(df_md5[["path", "md5"]], on="path", how="left")

    return df


# ========================= Stage B =========================


def run_stage_b(
    root: Path,
    df_a: pd.DataFrame,
    fast_no_identity: bool,
    similarity_treshold=0.65,
    yolo_batch_size=32,
) -> pd.DataFrame:

    # Embedder only if we will run identity
    embedder = None
    if not fast_no_identity:
        embedder = get_facenet_embedder(DEVICE)
        print("[INFO] Face embedder (facenet-pytorch) ready on", DEV_STR)

    # Filter: sharp images with person_count >= 1 (OPTIMIZED)
    sharp_with_person = df_a[
        (df_a["focus_label"] == "subject_in_focus") & (df_a["person_count"] >= 1)
    ].copy()

    print(
        f"[INFO] Stage B: Face detection on {len(sharp_with_person)} images (sharp + person_count >= 1)"
    )

    # 1) Try YOLOv8-Face first
    yolo_enabled, yolo_face_model = build_yolov8_face("./yolov8n-face-lindevs.pt")

    # 2) Initialize legacy detectors only if YOLO is NOT available
    rf_enabled = False
    rf_detect_fn = None
    rf_model = None
    mtcnn = None
    mtcnn_cpu = None

    if not yolo_enabled:
        log.info("YOLOv8-Face not available. Falling back to RetinaFace + MTCNN detectors.")

        # RetinaFace
        rf_enabled, rf_detect_fn, rf_model = build_retinaface()
        if not rf_enabled:
            log.warning("RetinaFace build failed; will rely on MTCNN only.")

        # MTCNN (only build if facenet_pytorch is installed / usable)
        try:
            from facenet_pytorch import MTCNN

            mtcnn_device = DEVICE
            mtcnn_cpu = None

            if DEV_STR == "mps":
                log.info(
                    "MTCNN primary detector forced to CPU due to MPS adaptive pooling limitation."
                )
                mtcnn_device = torch.device("cpu")

            mtcnn = MTCNN(keep_all=True, device=mtcnn_device)

            if mtcnn_device.type != "cpu":
                # Backup CPU MTCNN in case of MPS bugs
                mtcnn_cpu = MTCNN(keep_all=True, device=torch.device("cpu"))

        except Exception as e:
            log.warning("MTCNN initialization failed: %s", e)
            mtcnn = None
            mtcnn_cpu = None

    else:
        log.info("YOLOv8-Face available; skipping RetinaFace + MTCNN initialization.")

    for idx in tqdm(sharp_with_person.index, desc="Stage B: face-detect", unit="img"):
        row = df_a.loc[idx]
        p = root / row["path"]
        bgr = load_bgr_exif_safe(p)
        if bgr is None:
            continue

        person_count = int(row["person_count"])

        boxes, lms = detect_faces_smart(
            bgr,
            person_count,
            rf_enabled,
            rf_detect_fn,
            rf_model,
            mtcnn,
            mtcnn_cpu,
            yolo_face_model=yolo_face_model if yolo_enabled else None,
            yolo_batch_size=yolo_batch_size,
        )
        boxes_py = to_box_list_strict(boxes)
        df_a.at[idx, "_face_boxes"] = boxes_py
        df_a.at[idx, "num_faces_found"] = int(len(boxes_py))

    # Identity: ONLY for single-person + faces >= 1 (OPTIMIZED)
    if not fast_no_identity:
        single_person_with_faces = df_a[
            (df_a["focus_label"] == "subject_in_focus")
            & (df_a["person_count"] >= 1)
            & (df_a["num_faces_found"] == 1)
        ].copy()

        print(
            f"[INFO] Stage B: Identity clustering on {len(single_person_with_faces)} images (single-person + faces)"
        )

        face_vecs = []
        img_map = []

        for idx in tqdm(single_person_with_faces.index, desc="Stage B: face-embed", unit="img"):
            row = df_a.loc[idx]
            p = root / row["path"]
            bgr = load_bgr_exif_safe(p)
            boxes = row["_face_boxes"]

            if not boxes or bgr is None:
                continue

            E = embed_faces_batch(embedder, bgr, boxes)
            if E is None:
                continue

            for k in range(E.shape[0]):
                face_vecs.append(E[k])
                img_map.append((idx, k))

        if face_vecs:
            E = np.vstack(face_vecs).astype(np.float64, copy=False)

            labels = graph_clusters(E, sim_thresh=similarity_treshold)

            labels = remap_labels_sequential(labels)

            # Assign per-image labels
            per_img_labels: Dict[int, List[int]] = {}
            for lab, (img_idx, _k) in zip(labels, img_map):
                per_img_labels.setdefault(img_idx, []).append(int(lab))

            for img_idx, labs in per_img_labels.items():
                df_a.at[img_idx, "_face_id_labels"] = labs
                valid = [l for l in labs if l >= 0]
                df_a.at[img_idx, "identity_group"] = (
                    int(np.bincount(valid).argmax()) if valid else -1
                )

    return df_a


# ========================= Stage C =========================


def run_stage_c(root: Path, df_b: pd.DataFrame, openclip_batch_size=32) -> pd.DataFrame:
    # Landscape track: sharp & person_count == 0 (OPTIMIZED)
    land_df = df_b[(df_b["focus_label"] == "subject_in_focus") & (df_b["person_count"] == 0)].copy()

    land_paths = [root / row["path"] for _, row in land_df.iterrows()]

    if not land_paths:
        print("[INFO] Stage C: No landscape images to process")
        return df_b

    print(f"[INFO] Stage C: Processing {len(land_paths)} landscape images")

    model, preprocess = get_openclip(DEVICE)
    embs = embed_images_openclip_batch(
        model, preprocess, DEVICE, land_paths, batch_size=openclip_batch_size
    )

    if not embs:
        return df_b

    names = list(embs.keys())
    E = np.vstack([embs[n] for n in names]).astype(np.float64, copy=False)
    labels = cluster_hdbscan(E, min_cluster_size=3)

    # Update scene_group for landscape images
    for name, lab in zip(names, labels):
        mask = df_b["path"] == name
        df_b.loc[mask, "scene_group"] = int(lab)

    return df_b


# ========================= Final Label =========================


def compute_id_image_counts(df) -> Mapping[int, int]:
    """
    For each person_id, count in how many *images* it appears.
    Assumes df has a column '_face_id_labels' (list of ints per row).
    """
    id_image_counts = Counter()

    for _, row in df.iterrows():
        labels = row.get("_face_id_labels", [])
        if not isinstance(labels, (list, tuple)):
            continue
        # Unique IDs within this image
        ids_in_image = {l for l in labels if isinstance(l, int) and l >= 0}
        for pid in ids_in_image:
            id_image_counts[pid] += 1

    return id_image_counts


def final_group_name(row, id_image_counts: Mapping[int, int]) -> str:
    """
    Determine final folder name for an image, using global id_image_counts
    so that we only create Person_{id} folders when there are >= 2 images
    for that identity.
    """
    # 1) Blur / soft → everything goes into a single bucket
    if row["focus_label"] != "subject_in_focus":
        return "00_Blurred_or_LowQuality"

    person_count = int(row.get("person_count", 0))
    faces = int(row.get("num_faces_found", 0))

    # 2) No people detected → landscape / scene branch
    if person_count == 0:
        sg = int(row.get("scene_group", -100))
        if sg >= 0:
            return f"20_Landscape_Scenes/Scene_{sg}"
        if sg == -1:
            return "20_Landscape_Scenes/01_Unique_Scene_Noise"
        return "30_Outliers/02_Unclassified_Sharp_NoFace"

    # 3) Group / large-group decisions based on faces,
    #    so person_count>=2 but faces==1 still falls through to identity logic.
    if 2 <= faces <= 5:
        return "10_People/Group"

    if faces > 5:
        return "10_People/98_General_Large_Group"

    # 4) Single-face / silhouettes / identity classification

    # faces == 0 → silhouettes / backs (even if person_count > 1)
    if faces == 0:
        return "10_People/96_Silhouette_or_Back"

    # faces >= 1 → identity classification using clustering labels
    labels = row.get("_face_id_labels", [])
    valid_ids = sorted(set([l for l in labels if isinstance(l, int) and l >= 0]))

    # Helper to decide person folder vs singleton bucket
    def _person_or_singleton(pid: int) -> str:
        count = int(id_image_counts.get(pid, 0))
        if count >= 2:
            return f"10_People/Person_{pid}"
        else:
            # All identities that only appear in a single image go here
            return "10_People/95_Single_Identities"

    # Exactly one identity → clean assignment, but check global count
    if len(valid_ids) == 1:
        return _person_or_singleton(valid_ids[0])

    # Multiple identities present → pick a clear winner if there is one
    if len(valid_ids) > 1:
        counts = Counter([l for l in labels if isinstance(l, int) and l >= 0])
        if counts:
            top_id, top_cnt = counts.most_common(1)[0]
            # Ensure there isn't a tie for top count
            if list(counts.values()).count(top_cnt) == 1:
                return _person_or_singleton(top_id)
        # Ambiguous / noisy mixture of faces
        return "10_People/97_Unidentified_Noise"

    # No valid face IDs at all despite faces >= 1 → treat as noise
    return "10_People/97_Unidentified_Noise"


# ========================= CLI =========================


def parse_args():
    ap = argparse.ArgumentParser(description="PicSort v9 - MPS optimized with batch processing")
    ap.add_argument("--root", type=str, required=True, help="Folder of images")
    ap.add_argument("--limit", type=int, default=100000, help="Max images to process")

    # Processing options
    ap.add_argument(
        "--fast_no_identity", type=int, default=0, help="Skip identity clustering (0/1)"
    )

    # Thresholds
    ap.add_argument("--t_subj", type=float, default=20.0, help="Subject sharpness threshold")
    ap.add_argument("--t_bg", type=float, default=90.0, help="Background sharpness threshold")
    ap.add_argument("--closeness", type=float, default=25.0, help="Focus closeness threshold")

    # Person detector tuning
    ap.add_argument(
        "--person_conf", type=float, default=0.50, help="YOLO person confidence threshold"
    )
    ap.add_argument(
        "--person_min_area", type=float, default=0.01, help="YOLO person min area ratio"
    )

    # Batch sizes
    ap.add_argument("--yolo_batch", type=int, default=32, help="YOLO batch size")
    ap.add_argument("--openclip_batch", type=int, default=32, help="OpenCLIP batch size")

    # Face similarity treshold
    ap.add_argument("--sim_thres", type=float, default=0.65, help="Face similarity threshold")

    return ap.parse_args()


# ========================= Summaries =========================


def print_summaries(df: pd.DataFrame):
    print("\n[INFO] Final groups:")
    print(df["final_group_name"].value_counts().to_string())

    if "person_count" in df.columns:
        print("\n[INFO] People counts (all images):")
        print(df["person_count"].value_counts().sort_index().to_string())

    if "_face_id_labels" in df.columns:
        c = Counter()
        for labels in df["_face_id_labels"]:
            if isinstance(labels, list):
                c.update(set([l for l in labels if isinstance(l, int) and l >= 0]))
        if c:
            print("\n[INFO] Identity image coverage (images per person id):")
            for pid, cnt in sorted(c.items()):
                print(f"  Person_{pid}: {cnt}")

    # Duplicate statistics
    if "md5" in df.columns:
        n_unique = df["md5"].nunique()
        n_total = len(df)
        n_dupes = n_total - n_unique
        if n_dupes > 0:
            print(
                f"\n[INFO] Deduplication: {n_total} total images, {n_unique} unique, {n_dupes} duplicates"
            )


# ======================== Move =========================


def move(
    root: Union[str, Path],
    df_final: pd.DataFrame,
    path_col: str = "path",
    group_col: str = "final_group_name",
    dry_run: bool = False,
) -> None:
    """
    Create folders under `root` based on `final_group_name` and move images there.

    - root: base directory containing the original images.
    - df_final: DataFrame with at least:
        * path_col: original image path (relative to root, or just filename).
        * group_col: target group path (e.g. "10_People/Group", "10_People/Person_1").
    - dry_run: if True, only print planned moves; don't actually move files.
    """
    root = Path(root)

    for _, row in df_final.iterrows():
        group_name = row.get(group_col)
        rel_path = row.get(path_col)

        # Skip rows with missing info
        if pd.isna(group_name) or pd.isna(rel_path):
            continue

        src = root / str(rel_path)

        # Skip if source doesn't exist (already moved or missing)
        if not src.exists():
            # You could log a warning here if you want
            # print(f"[WARN] Source missing, skipping: {src}")
            continue

        # final_group_name may contain subfolders like "10_People/Group"
        dst_dir = root / str(group_name)
        dst_dir.mkdir(parents=True, exist_ok=True)

        dst = dst_dir / src.name

        if dry_run:
            print(f"[DRY RUN] {src} -> {dst}")
            continue

        # Avoid overwriting silently; if collision, you can choose to rename or skip
        if dst.exists():
            # Example: append a counter to avoid collision
            stem = dst.stem
            suffix = dst.suffix
            counter = 1
            new_dst = dst_dir / f"{stem}_{counter}{suffix}"
            while new_dst.exists():
                counter += 1
                new_dst = dst_dir / f"{stem}_{counter}{suffix}"
            dst = new_dst

        shutil.move(str(src), str(dst))


# ========================= Main =========================


def main():
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    print(f"[INFO] Scanning: {root}")
    if not root.exists():
        print(f"[ERROR] Folder not found: {root}")
        return

    imgs = list_images(root)
    if not imgs:
        print("[INFO] No images found.")
        return
    imgs = imgs[: args.limit]
    print(f"[INFO] Found {len(imgs)} images")
    print(f"[INFO] Device: {DEV_STR}")

    # MD5 calculation first
    df_md5 = calculate_md5_batch(imgs)

    # Stage A (process unique images only)
    dfA = run_stage_a(
        root,
        imgs,
        df_md5,
        t_subj=args.t_subj,
        t_bg=args.t_bg,
        closeness=args.closeness,
        person_conf=args.person_conf,
        person_min_area=args.person_min_area,
        yolo_batch_size=args.yolo_batch,
    )

    # Stage B
    dfB = run_stage_b(
        root,
        dfA.copy(),
        fast_no_identity=bool(args.fast_no_identity),
        similarity_treshold=args.sim_thres,
        yolo_batch_size=args.yolo_batch,
    )

    # Stage C
    dfC = run_stage_c(root, dfB.copy(), openclip_batch_size=args.openclip_batch)

    # Final naming
    # dfC["final_group_name"] = dfC.apply(final_group_name, axis=1)
    id_image_counts = compute_id_image_counts(dfC)
    dfC["final_group_name"] = dfC.apply(lambda row: final_group_name(row, id_image_counts), axis=1)

    # Broadcast to duplicates
    df_final = broadcast_to_duplicates(dfC, df_md5)

    # df_final = dfC
    move(root, df_final)

    df_sorted = df_final.sort_values(by="final_group_name")

    # Save results
    out_csv = root / "picsort_results.csv"
    df_sorted.to_csv(out_csv, index=False)
    print(f"[INFO] Saved results to: {out_csv}")
    log.info("Saved results to: %s", out_csv)

    print_summaries(df_sorted)
    log.debug("Run complete. See picsort_pipeline.log for details.")


if __name__ == "__main__":
    main()
