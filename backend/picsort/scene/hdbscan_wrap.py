import hdbscan
import numpy as np
from picsort.config import AppConfig


def cluster_hdbscan(E: np.ndarray, cfg: AppConfig) -> np.ndarray:
    """Cluster L2-normalized embeddings with HDBSCAN on Euclidean distance

    Args:
        E (np.ndarray): Embeddings
        cfg (AppConfig): App config

    Returns:
        np.ndarray: Cluster labels
    """
    n = E.shape[0]
    if n < cfg.scene.hdbscan_min_cluster_size:
        return np.full((n,), -1, dtype=int)

    E64 = E.astype(np.float64, copy=False)

    cluster = hdbscan.HDBSCAN(
        min_cluster_size=cfg.scene.hdbscan_min_cluster_size,
        metric="euclidean",
        cluster_selection_method="eom",
        core_dist_n_jobs=1,
    )

    labels = cluster.fit_predict(E64)
    return labels
