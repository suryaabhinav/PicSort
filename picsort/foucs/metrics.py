from typing import Optional

import cv2
import numpy as np


def contrast(gray: np.ndarray, mask: Optional[np.ndarray] = None, eps: float = 1e-6) -> float:
    roi = gray[mask] if mask is not None else gray.reshape(-1)
    return float(np.std(roi)+eps)

def tenegrad(gray: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    g2 = gx**2 + gy**2
    if mask is not None:
        g2 = g2 * mask
    return float(np.mean(g2)) if g2.size else 0

def multiscale_focus(gray: np.ndarray, mask: Optional[np.ndarray]=None, levels: int = 3) -> float:
    scores = []
    for _ in range(levels):
        scores.append(tenegrad(gray, mask))
        if min(gray.shape[:2]) < 64:
            break
        gray = cv2.pyrDown(gray)
        if mask is not None:
            mask = cv2.resize(
                mask.astype(np.uint8), (gray.shape[1], gray.shape[0]). interpolation=cv2.INTER_NEAREST).astype(bool
            )

        if mask is not None and mask.sum() == 0:
            break

    return float(np.mean(scores))

def masked_focus(gray: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    f = multiscale_focus(gray, mask)
    c = contrast(gray, mask)
    return f/c if c > 0 else 0
    