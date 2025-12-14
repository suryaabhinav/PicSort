from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from picsort.config import RuntimeContext

TORCH_INFERENCE = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad


def get_facenet_embedder(ctx: RuntimeContext) -> object:
    """Facenet embedder loader

    Args:
        ctx (RuntimeContext): Runtime context

    Returns:
        object: Facenet embedder
    """
    from facenet_pytorch import InceptionResnetV1

    model = InceptionResnetV1(pretrained="vggface2").eval().to(ctx.device_str)
    return model


def embed_face_batch(
    embedder: object, bgr: np.ndarray, boxes: List[Tuple[int, int, int, int]]
) -> Optional[np.ndarray]:
    """Embed multiple faces from single image

    Args:
        embedder (object): Embedder
        bgr (np.ndarray): Image
        boxes (List[Tuple[int, int, int, int]]): Bounding boxes of faces

    Returns:
        Optional[np.ndarray]: Embeddings of faces
    """
    if not boxes:
        return None
    import torchvision.transforms as T

    trans = T.Compose(
        [
            T.ToTensor(),
            T.Resize((160, 160), antialias=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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

    batch = torch.stack(rgbs).to(embedder.device, memory_format=torch.channels_last)
    with TORCH_INFERENCE():
        embeds = (
            embedder(batch.to(next(embedder.parameters()).device))
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    embeds /= np.linalg.norm(embeds, axis=1, keepdims=True) + 1e-12
    return embeds
