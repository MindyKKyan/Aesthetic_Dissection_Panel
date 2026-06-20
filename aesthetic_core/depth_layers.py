"""
Monocular depth estimation + foreground / midground / background layer separation.
Uses Intel DPT (MiDaS hybrid) via transformers — lazy-loaded on first analyze.
"""

from __future__ import annotations

import base64
import io

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .config import DEPTH_MODEL_ID, DEVICE

try:
    import cv2

    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

_depth_processor = None
_depth_model = None

_THUMB = 384


def _load_depth_model():
    global _depth_processor, _depth_model
    if _depth_model is not None:
        return _depth_processor, _depth_model
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    _depth_processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL_ID)
    _depth_model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_ID).to(DEVICE)
    _depth_model.eval()
    return _depth_processor, _depth_model


@torch.no_grad()
def _estimate_depth(img: Image.Image) -> np.ndarray:
    """Relative depth map; higher = closer to camera."""
    processor, model = _load_depth_model()
    inputs = processor(images=img.convert("RGB"), return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    outputs = model(**inputs)
    h, w = img.size[1], img.size[0]
    depth = F.interpolate(
        outputs.predicted_depth.unsqueeze(1),
        size=(h, w),
        mode="bicubic",
        align_corners=False,
    ).squeeze().cpu().numpy()
    return depth.astype(np.float32)


def _normalize_depth(depth: np.ndarray) -> np.ndarray:
    lo, hi = float(depth.min()), float(depth.max())
    if hi - lo < 1e-6:
        return np.zeros_like(depth)
    return (depth - lo) / (hi - lo)


def _feather_mask(mask: np.ndarray, sigma: float = 5.0) -> np.ndarray:
    m = mask.astype(np.float32)
    if _HAS_CV2:
        k = int(max(3, sigma * 4)) | 1
        return cv2.GaussianBlur(m, (k, k), sigma)
    return m


def _depth_colormap(depth_norm: np.ndarray) -> Image.Image:
    """Near = warm, far = cool."""
    h, w = depth_norm.shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    t = depth_norm
    canvas[:, :, 0] = (t * 255).astype(np.uint8)
    canvas[:, :, 1] = (np.sin(t * np.pi) * 180).astype(np.uint8)
    canvas[:, :, 2] = ((1 - t) * 220).astype(np.uint8)
    return Image.fromarray(canvas)


def _composite_layer(rgb: np.ndarray, mask: np.ndarray, bg: tuple[int, int, int] = (245, 245, 247)) -> Image.Image:
    m = np.clip(mask, 0, 1)[..., None]
    bg_arr = np.array(bg, dtype=np.float32)
    out = rgb.astype(np.float32) * m + bg_arr * (1 - m)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def _to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def separate_depth_layers(img: Image.Image) -> dict:
    """
    Return depth map + 3 separated layers and summary stats.
    Keys: depth_map, foreground, midground, background, stats (all except stats are data URLs).
    """
    img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > _THUMB:
        scale = _THUMB / max(w, h)
        work = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    else:
        work = img

    depth = _estimate_depth(work)
    depth_norm = _normalize_depth(depth)

    t_lo, t_hi = np.percentile(depth_norm, [38, 62])
    fg_mask = _feather_mask((depth_norm >= t_hi).astype(np.float32))
    bg_mask = _feather_mask((depth_norm <= t_lo).astype(np.float32))
    mg_mask = _feather_mask(
        np.clip(1.0 - np.maximum(fg_mask, bg_mask), 0, 1).astype(np.float32)
    )

    rgb = np.array(work, dtype=np.float32)
    depth_vis = _depth_colormap(depth_norm)

    fg_cov = float((depth_norm >= t_hi).mean())
    mg_cov = float(((depth_norm > t_lo) & (depth_norm < t_hi)).mean())
    bg_cov = float((depth_norm <= t_lo).mean())
    depth_spread = float(depth_norm.std())

    return {
        "depth_map": _to_data_url(depth_vis),
        "foreground": _to_data_url(_composite_layer(rgb, fg_mask)),
        "midground": _to_data_url(_composite_layer(rgb, mg_mask)),
        "background": _to_data_url(_composite_layer(rgb, bg_mask)),
        "stats": {
            "foreground_coverage": round(fg_cov, 3),
            "midground_coverage": round(mg_cov, 3),
            "background_coverage": round(bg_cov, 3),
            "depth_spread": round(depth_spread, 3),
            "model": DEPTH_MODEL_ID,
            "note": "Brighter depth = nearer. Layers split by relative depth percentiles.",
        },
    }
