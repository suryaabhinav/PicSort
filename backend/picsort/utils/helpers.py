import shutil
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd


def rot90_ccw(bgr: np.ndarray) -> np.ndarray:
    """Rotate image 90 degrees counter-clockwise"""
    return np.ascontiguousarray(np.rot90(bgr, k=1))


def rot90_cw(bgr: np.ndarray) -> np.ndarray:
    """Rotate image 90 degrees clockwise"""
    return np.ascontiguousarray(np.rot90(bgr, k=3))


def to_box_list_strict(boxes: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Return pure-python list of (x1, y1, x2, y2)"""
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
    except Exception as e:
        out = []
        for b in boxes:
            if isinstance(b, (list, tuple)) and len(b) >= 4:
                x1, y1, x2, y2 = map(int, b[:4])
                if x2 > x1 and y2 > y1:
                    out.append((x1, y1, x2, y2))
        return out


def apply_move(
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

        if pd.isna(group_name) or pd.isna(rel_path):
            continue

        src = root / str(rel_path)

        if not src.exists():
            continue

        dst_dir = root / str(group_name)
        dst_dir.mkdir(parents=True, exist_ok=True)

        dst = dst_dir / src.name

        if dry_run:
            print(f"[DRY RUN] {src} -> {dst}")
            continue

        if dst.exists():
            stem = dst.stem
            suffix = dst.suffix
            counter = 1
            new_dst = dst_dir / f"{stem}_{counter}{suffix}"
            while new_dst.exists():
                counter += 1
                new_dst = dst_dir / f"{stem}_{counter}{suffix}"
            dst = new_dst

        shutil.move(str(src), str(dst))
