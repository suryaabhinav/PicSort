import torch

from backend.picsort.config import RuntimeContext


def choose_device(preferred: str) -> RuntimeContext:
    if preferred == "mps" and torch.backends.mps.is_available():
        return RuntimeContext(device_str="mps")
    if preferred == "cuda" and torch.cuda.is_available():
        return RuntimeContext(device_str="cuda")
    return RuntimeContext(device_str="cpu")
