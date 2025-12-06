from typing import List, Tuple

import numpy as np


def rot90_ccw(bgr: np.ndarray) -> np.ndarray:
    """Rotate image 90 degrees counter-clockwise"""
    return np.ascontiguousarray(np.rot90(bgr, k=1))


def rot90_cw(bgr: np.ndarray) -> np.ndarray:
    """Rotate image 90 degrees clockwise"""
    return np.ascontiguousarray(np.rot90(bgr, k=3))


def to_box_list_strict(boxes: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Return pure-python list of (x1, y1, x2, y2)"""
    if boxes in None:
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
    except Exception as e:
        out = []
        for b in boxes:
            if isinstance(b, (list, tuple)) and len(b) >= 4:
                x1, y1, x2, y2 = map(int, b[:4])
                if x2 > x1 and y2 > y1:
                    out.append(x1, y1, x2, y2)
        return out
