"""Choose the best supported PyTorch device for this project."""

import torch


def mps_is_available() -> bool:
    """Return whether this PyTorch build can use Apple's MPS backend."""
    mps_backend = getattr(torch.backends, "mps", None)
    return bool(mps_backend and mps_backend.is_built() and mps_backend.is_available())


def get_device(verbose: bool = True) -> torch.device:
    """Prefer CUDA, then Apple MPS, then the always-available CPU backend."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif mps_is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if verbose:
        print(f"Using device: {device}")
    return device
