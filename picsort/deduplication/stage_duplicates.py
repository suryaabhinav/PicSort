from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from picsort.config import AppConfig
from picsort.io.utils import list_images, md5_hash
from picsort.pipeline.orchestrator import log

ProgressFn = Optional[Callable[[int, int, Optional[str]], None]]


def stage_duplicates(root: Path, cfg: AppConfig, progress: ProgressFn = None) -> pd.DataFrame:
    """Stage Duplicates: Calculate MD5 hashes for all the images and find duplicates

    Args:
        root (Path): Root directory of images
        cfg (AppConfig): Configuration object
        progress (ProgressFn, optional): Progress callback function. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame of stage duplicates results
    """
    image_paths = list_images(root)
    rows = []
    for p in image_paths:
        try:
            md5 = md5_hash(p)
            rows.append({"path": str(p), "md5": md5})
        except Exception as e:
            log.warning(f"Failed to calculate MD5 for {p}: {e}")
            continue

    df_stage_duplicates = pd.DataFrame(rows)
    df_stage_duplicates["is_duplicate"] = df_stage_duplicates.duplicated(subset="md5", keep="first")
    n_dupes = df_stage_duplicates["is_duplicate"].sum()

    if n_dupes > 0:
        log.info(f"Found {n_dupes} duplicates")
        log.info(f"Will process {len(df) - n_dupes} unique images")
    else:
        log.info("No duplicates found")

    return df_stage_duplicates
