"""Project-local config — re-exports shared core + Gradio launch flags."""

import os

from aesthetic_core.config import DEVICE, HF_TOKEN, OPENAI_API_KEY

SHARE_LOCAL = os.environ.get("DV_SHARE", "0") == "1"

__all__ = ["DEVICE", "HF_TOKEN", "OPENAI_API_KEY", "SHARE_LOCAL"]
