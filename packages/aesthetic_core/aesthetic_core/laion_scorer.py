"""
LAION Aesthetic Predictor V1
CLIP ViT-L/14 (768-d CLS embedding) + sa_0_4_vit_l_14_linear.pth → score ~0–10
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image

from .config import CLIP_MODEL_ID, DEVICE, LAION_WEIGHTS_PATH, LAION_WEIGHTS_URL

_clip_model = None
_clip_processor = None
_aesthetic_head: nn.Linear | None = None


def _download_weights() -> Path:
    if not LAION_WEIGHTS_PATH.exists():
        LAION_WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(LAION_WEIGHTS_URL, LAION_WEIGHTS_PATH)
    return Path(LAION_WEIGHTS_PATH)


def _load_clip():
    global _clip_model, _clip_processor
    if _clip_model is not None:
        return _clip_model, _clip_processor
    from transformers import CLIPModel, CLIPProcessor

    _clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(DEVICE)
    _clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    _clip_model.eval()
    return _clip_model, _clip_processor


def _load_aesthetic_head() -> nn.Linear:
    global _aesthetic_head
    if _aesthetic_head is not None:
        return _aesthetic_head
    _download_weights()
    head = nn.Linear(768, 1)
    state = torch.load(LAION_WEIGHTS_PATH, map_location="cpu", weights_only=True)
    head.load_state_dict(state)
    head.to(DEVICE).eval()
    _aesthetic_head = head
    return head


@torch.no_grad()
def clip_embedding_768(img: Image.Image) -> torch.Tensor:
    """Normalized 768-d CLIP ViT-L/14 image embedding (LAION convention)."""
    model, processor = _load_clip()
    inp = processor(images=img.convert("RGB"), return_tensors="pt")
    pixel_values = inp["pixel_values"].to(DEVICE)
    vision_out = model.vision_model(pixel_values=pixel_values)
    cls = vision_out.last_hidden_state[:, 0, :]
    emb = model.visual_projection(cls)
    return emb / emb.norm(dim=-1, keepdim=True)


@torch.no_grad()
def laion_aesthetic_score(img: Image.Image) -> float:
    """Return LAION aesthetic score clamped to 0–10."""
    emb = clip_embedding_768(img)
    head = _load_aesthetic_head()
    raw = float(head(emb).squeeze().item())
    return round(max(0.0, min(10.0, raw)), 2)
