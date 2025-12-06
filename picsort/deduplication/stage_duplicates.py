from pathlib import Path

import pandas as pd

from picsort.config import AppConfig
from picsort.io.utils import list_images, md5_hash
from picsort.pipeline.orchestrator import log


def stage_duplicates(root: Path, cfg: AppConfig, progress=None) -> pd.DataFrame:
    """
    Calculate MD5 hashes for all the images and find duplicates
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

    df = pd.DataFrame(rows)
    df["is_duplicate"] = df.duplicated(subset="md5", keep="first")
    n_dupes = df["is_duplicate"].sum()

    if n_dupes > 0:
        log.info(f"Found {n_dupes} duplicates")
        log.info(f"Will process {len(df) - n_dupes} unique images")
    else:
        log.info("No duplicates found")

    return df
