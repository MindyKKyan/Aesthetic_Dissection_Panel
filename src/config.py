import os
import torch
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = BASE_DIR / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

# Device: MPS → CUDA → CPU (never hard-coded)
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

CLIP_MODEL_ID = "openai/clip-vit-large-patch14"
LAION_WEIGHTS_URL = (
    "https://github.com/LAION-AI/aesthetic-predictor/raw/main/sa_0_4_vit_l_14_linear.pth"
)
LAION_WEIGHTS_PATH = WEIGHTS_DIR / "sa_0_4_vit_l_14_linear.pth"

SHARE_LOCAL = os.environ.get("DV_SHARE", "0") == "1"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN", os.environ.get("HUGGINGFACEHUB_API_TOKEN", ""))
