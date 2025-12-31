import hashlib
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from api.logging_config import log
from PIL import Image, ImageOps

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".gif"}


def list_images(root: Path) -> List[Path]:
    """List all images in the given directory"""
    return [
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in VALID_EXTS and not p.name.startswith("._")
    ]


def md5_hash(path: Path) -> str:
    """Calculate MD5 hash of a file"""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_pil_exif_rgb(path: Path) -> Image.Image:
    """Load image from path with EXIF orientation"""
    im = Image.open(path)
    im = ImageOps.exif_transpose(im).convert("RGB")
    return im


def load_bgr_exif_safe(path: Path) -> Optional[np.ndarray]:
    """Load image from path with EXIF orientation and convert to BGR

    Args:
        path (Path): Path to image

    Returns:
        Optional[np.ndarray]: BGR image array
    """
    try:
        pil = load_pil_exif_rgb(path)
        arr = np.array(pil)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except FileNotFoundError:
        log.warning(f"File not found: {path}")
        return None
    except Exception:
        try:
            arr = np.fromfile(str(path), dtype=np.uint8)
        except FileNotFoundError:
            log.warning(f"Image missing on disk: {path}")
            return None
        if arr.size == 0:
            log.warning(f"Empty image: {path}")
            return None
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            log.warning(f"Failed to decode image: {path}")
            return None
        return bgr
