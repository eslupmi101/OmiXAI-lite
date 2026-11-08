"""Device selection shared by local runners and the Python API."""

import torch


def select_device(requested: str = "auto") -> torch.device:
    if requested not in {"auto", "cpu", "mps", "cuda"}:
        raise ValueError(f"Unknown device: {requested}")
    if requested == "auto":
        requested = ("cuda" if torch.cuda.is_available() else
                     "mps" if torch.backends.mps.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    return torch.device(requested)
