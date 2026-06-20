import os
from pathlib import Path

import torch

_PKG_DIR = Path(__file__).resolve().parent
_DEFAULT_WEIGHTS = _PKG_DIR.parent / "weights"
WEIGHTS_DIR = Path(os.environ.get("AESTHETIC_CORE_WEIGHTS_DIR", _DEFAULT_WEIGHTS))
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

CLIP_MODEL_ID = "openai/clip-vit-large-patch14"
DEPTH_MODEL_ID = "Intel/dpt-hybrid-midas"
LAION_WEIGHTS_URL = (
    "https://github.com/LAION-AI/aesthetic-predictor/raw/main/sa_0_4_vit_l_14_linear.pth"
)
LAION_WEIGHTS_PATH = WEIGHTS_DIR / "sa_0_4_vit_l_14_linear.pth"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")


def resolve_hf_token() -> str:
    """HF token from env vars, then `hf auth login` cache."""
    for key in ("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(key, "").strip()
        if value:
            return value
    try:
        from huggingface_hub import get_token

        cached = get_token()
        if cached:
            return cached
    except Exception:
        pass
    return ""


HF_TOKEN = resolve_hf_token()
