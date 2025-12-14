from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import PIL.Image
import torch
import torch.nn as nn

from api.logging_config import log
from picsort.config import AppConfig, RuntimeContext
from picsort.io.utils import load_pil_exif_rgb

TORCH_INFERENCE = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad


def get_openclip(
    ctx: RuntimeContext,
) -> Tuple[nn.Module, Callable[[PIL.Image.Image], torch.Tensor]]:
    """OpenCLIP Loader

    Args:
        ctx (RuntimeContext): Runtime Context

    Returns:
        Tuple[nn.Module, Callable[[PIL.Image.Image], torch.Tensor]]: OpenCLIP model and preprocess function
    """
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name="ViT-B-16", pretrained="laion2b_s34b_b88k", device=ctx.device_str
    )
    log.info(f"[INFO] OpenCLIP ViT-B/16 (LAION2B) loaded on {ctx.device_str}")
    return model, preprocess


def embed_images_openclip_batch(
    model: nn.Module,
    preprocess: Callable[[PIL.Image.Image], torch.Tensor],
    paths: List[Path],
    cfg: AppConfig,
    ctx: RuntimeContext,
) -> Dict[str, np.ndarray]:
    """Batch embed clips using OpenCLIP

    Args:
        model (nn.Module): OpenCLIP model
        preprocess (Callable[[PIL.Image.Image], torch.Tensor]): Preprocess function
        paths (List[Path]): Image paths
        cfg (AppConfig): App config
        ctx (RuntimeContext): Runtime context

    Returns:
        Dict[str, np.ndarray]: Embeddings
    """
    embeds = {}

    for i in range(0, len(paths), cfg.scene.batch_size):
        batch_paths = paths[i : i + cfg.scene.batch_size]
        batch_tensors = []
        vaild_paths = []

        for p in batch_paths:
            try:
                pil = load_pil_exif_rgb(p)
                im = preprocess(pil).unsqueeze(0)
                batch_tensors.append(im)
                vaild_paths.append(p)
            except Exception as e:
                log.warning(f"[WARNING] OpenCLIP preprocess failed on {p.name}: {e}")

        if batch_tensors:
            try:
                batch = torch.cat(batch_tensors, dim=0).to(ctx.device_str)
                with TORCH_INFERENCE():
                    feature = model.encode_image(batch)
                feature_np = feature.detach().cpu().numpy().astype(np.float32)

                for p, v in zip(vaild_paths, feature_np):
                    embeds[p.name] = v.reshape(-1)
            except Exception as e:
                log.warning(f"[WARNING] OpenCLIP embed failed on {p.name}: {e}")

    return embeds
