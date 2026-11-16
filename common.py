"""Shared constants and helpers for the face-mask detection project."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Keep every cache/config inside the project folder (important when C: is nearly full).
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".yolo_config"))
os.environ.setdefault("KAGGLEHUB_CACHE", str(ROOT / "datasets" / ".kagglehub_cache"))

# Class order used for training AND inference. Do not change after training.
CLASS_NAMES = ["with_mask", "mask_weared_incorrect", "without_mask"]

# BGR colours used when drawing.
COLOR_BY_NAME = {
    "with_mask": (0, 200, 0),               # green  -> mask worn correctly
    "mask_weared_incorrect": (0, 165, 255), # orange -> mask worn incorrectly
    "without_mask": (0, 0, 230),            # red    -> no mask
}
DEFAULT_COLOR = (200, 200, 200)

PRETTY = {
    "with_mask": "Mask OK",
    "mask_weared_incorrect": "Mask INCORRECT",
    "without_mask": "NO MASK",
}

DEFAULT_WEIGHTS = ROOT / "weights" / "best.pt"
DEFAULT_DATA_YAML = ROOT / "datasets" / "face_mask" / "data.yaml"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm", ".m4v"}


def pick_device(requested: str = "auto") -> str:
    """Return a device string understood by ultralytics ('0', 'mps' or 'cpu')."""
    if requested and requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "0"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def is_cuda(device: str) -> bool:
    return device not in ("cpu", "mps") and (str(device).replace(",", "").isdigit() or str(device).startswith("cuda"))
