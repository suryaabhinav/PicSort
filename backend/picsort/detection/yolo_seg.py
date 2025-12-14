from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from ultralytics import YOLO

from backend.api.logging_config import log
from backend.picsort.config import AppConfig, Models, RuntimeContext
from backend.picsort.io.utils import load_bgr_exif_safe


class YOLOProcessor:
    """
    Unified YOLO porcessor for segmentation and person detection with batch support
    """

    def __init__(self, cfg: AppConfig, ctx: RuntimeContext, models: Models):
        self.seg_ok = False
        self.person_conf = cfg.yolo.person_conf
        self.seg_conf = cfg.yolo.seg_conf
        self.imgsz = cfg.yolo.imgsz
        self.batch_size = cfg.yolo.batch_size
        self.device = ctx.device_str
        self.seg_model_path = models.yolo_seg_model

        if models.yolo_seg_model is None:
            models.yolo_seg_model = "yolo11m-seg.pt"
            self.seg_model_path = models.yolo_seg_model

        try:
            self.seg_model = YOLO(self.seg_model_path)
            if self.device in ["mps", "cuda"]:
                self.seg_model.to(self.device)
            try:
                self.seg_model.fuse()
            except Exception as e:
                log.warning(f"Failed to fuse segmentation model: {e}")
            self.seg_ok = True
            log.info("[INFO] Segmentation model loaded successfully")
        except Exception as e:
            log.warning(f"Seg Init model failed: {e}")

    @staticmethod
    def _ellipse(bgr: np.ndarray) -> np.ndarray:
        """Fallback ellipse mask"""
        H, W = bgr.shape[:2]
        yy, xx = np.mgrid[0:H, 0:W]
        cx, cy = W / 2.0, H / 2.0
        rx, ry = W * 0.35, H * 0.35
        return ((xx - cx) ** 2) / (rx**2) + ((yy - cy) ** 2) / (ry**2) <= 1.0

    def _extract_mask(self, result) -> Optional[np.ndarray]:
        """Extract segmentation mask from YOLO model"""
        try:
            if result.masks is None or len(result.masks.data) == 0:
                return None

            masks = result.masks.data
            if hasattr(masks, "device") and masks.device.type != "cpu":
                masks = masks.cpu()
            masks = masks.numpy().astype(bool)

            orig_shape = result.orig_shape
            h, w = orig_shape[0], orig_shape[1]

            if masks.shape[1] != h or masks.shape[2] != w:
                resized = []
                for mask in masks:
                    mask8 = mask.astype(np.uint8) * 255
                    mask8 = cv2.resize(mask8, (w, h), interpolation=cv2.INTER_NEAREST)
                    resized.append(mask8.astype(bool))
                masks = np.stack(resized, axis=0)

            combined = np.logical_or.reduce(masks)
            return combined
        except Exception as e:
            log.warning(f"Failed to extract mask: {e}")
            return None

    def process_batch(self, image_paths: List[Path], batch_size: int) -> Dict[str, Dict]:
        """Process images in batchs for maskes and person counts

        Args:
            image_paths (List[Path]): List of image paths
            batch_size (int): Batch size.

        Returns:
            Dict[str, Dict]: Dictionary of image paths and their masks and person counts
        """

        results = {}
        if not self.seg_ok:
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

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]

            if self.seg_ok:
                try:
                    seg_results = self.seg_model.predict(
                        source=[str(p) for p in batch_paths],
                        batch=batch_size,
                        imgsz=self.imgsz,
                        conf=self.seg_conf,
                        device=self.device,
                        verbose=False,
                        stream=False,
                    )

                    for path, result in zip(batch_paths, seg_results):
                        mask = self._extract_mask(result)

                        person_count = 0
                        boxes = result.boxes
                        if boxes is not None and len(boxes) > 0:
                            cls = boxes.cls.cpu().numpy().astype(int)
                            conf = boxes.conf.cpu().numpy()

                            names = result.names

                            person_ids = [k for k, v in names.items() if v == "person"]
                            if person_ids:
                                pid = person_ids[0]
                                is_person = cls == pid
                                is_confident = conf >= self.person_conf
                                keep = is_person & is_confident
                                person_count = int(keep.sum())

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
                            "person_count": person_count,
                            "mask_type": mask_type,
                        }
                except Exception as e:
                    log.warning(f"[WARNING] YOLO seg batch failed: {e}")
                    for path in batch_paths:
                        bgr = load_bgr_exif_safe(path)
                        if bgr is not None:
                            result[path.name] = {
                                "mask": self._ellipse(bgr),
                                "person_count": 0,
                                "mask_type": "ellipse",
                            }

            else:
                for path in batch_paths:
                    bgr = load_bgr_exif_safe(path)
                    if bgr is not None:
                        results[path.name] = {
                            "mask": self._ellipse(bgr),
                            "person_count": 0,
                            "mask_type": "ellipse",
                        }

        return results
