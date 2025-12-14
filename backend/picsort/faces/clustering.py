import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def graph_clusters(E: np.ndarray, similarity_treshold: float) -> np.ndarray:
    """Graph cluster image embeddings

    Args:
        E (np.ndarray): Embeddings
        similarity_treshold (float): Similarity treshold

    Returns:
        np.ndarray: Cluster labels
    """
    n = E.shape[0]
    if n == 0:
        return np.array([], dtype=int)

    E64 = E.astype(np.float64, copy=False)
    similarity = cosine_similarity(E64)
    np.fill_diagonal(similarity, 1.0)
    adj = (similarity > similarity_treshold).astype(np.uint8)
    labels = -np.ones(n, dtype=int)
    cid = 0
    for i in range(n):
        if labels[i] != -1:
            continue
        stack = [i]
        labels[i] = cid
        while stack:
            u = stack.pop()
            neighbor = np.where(adj[u] == 1)[0]
            for v in neighbor:
                if labels[v] == -1:
                    labels[v] = cid
                    stack.append(v)

        cid += 1

    return labels


def remap_labels_sequential(labels: np.ndarray) -> np.ndarray:
    """Remap labels to sequential values

    Args:
        labels (np.ndarray): Labels

    Returns:
        np.ndarray: Sequential labels
    """
    mapping = {}
    nxt = 0
    out = labels.copy()
    for i, label in enumerate(labels):
        if label >= 0:
            if label not in mapping:
                mapping[label] = nxt
                nxt += 1
            out[i] = mapping[label]

    return out
