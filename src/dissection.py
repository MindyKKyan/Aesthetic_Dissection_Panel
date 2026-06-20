"""
Dissection proxies — defensible, computable aesthetic dimensions (0–1 bars).
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image

try:
    import cv2

    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


def _rgb_array(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"), dtype=np.float32) / 255.0


def _hsv_arrays(img: Image.Image):
    arr = _rgb_array(img)
    h, w, _ = arr.shape
    maxc = np.maximum(np.maximum(arr[:, :, 0], arr[:, :, 1]), arr[:, :, 2])
    minc = np.minimum(np.minimum(arr[:, :, 0], arr[:, :, 1]), arr[:, :, 2])
    delta = maxc - minc
    sat = delta / (maxc + 1e-6)
    hue = np.zeros_like(maxc)
    mask = delta > 1e-6
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    hue[mask & (maxc == r)] = ((g - b) / (delta + 1e-6))[mask & (maxc == r)] % 6
    hue[mask & (maxc == g)] = ((b - r) / (delta + 1e-6) + 2)[mask & (maxc == g)]
    hue[mask & (maxc == b)] = ((r - g) / (delta + 1e-6) + 4)[mask & (maxc == b)]
    hue = hue / 6.0
    val = maxc
    return hue, sat, val, arr


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _color_harmony(img: Image.Image) -> dict:
    """HSV hue histogram spread (entropy) × luminance contrast ratio proxy."""
    hue, sat, val, arr = _hsv_arrays(img)
    active = sat > 0.08
    if active.sum() < 10:
        spread = 0.3
    else:
        h_active = hue[active]
        hist, _ = np.histogram(h_active, bins=36, range=(0, 1), density=True)
        hist = hist + 1e-8
        entropy = -float(np.sum(hist * np.log(hist))) / math.log(36)
        spread = _clip01(entropy / 0.85)

    lum = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
    p95, p05 = float(np.percentile(lum, 95)), float(np.percentile(lum, 5))
    contrast = _clip01((p95 - p05) / 0.75)

    value = _clip01(0.55 * spread + 0.45 * contrast)
    return {
        "id": "color_harmony",
        "label": "Color Harmony",
        "value": round(value, 3),
        "raw": {
            "hue_entropy_norm": round(spread, 4),
            "luminance_contrast": round(contrast, 4),
        },
        "detail": f"hue spread {spread:.2f}, luminance contrast {contrast:.2f}",
    }


def _saliency_com(img: Image.Image) -> tuple[float, float, float]:
    """Return (cx, cy, offset) in 0–1 coords; offset = distance from center."""
    arr = _rgb_array(img)
    h, w, _ = arr.shape

    if _HAS_CV2:
        bgr = (arr * 255).astype(np.uint8)[:, :, ::-1]
        try:
            sal = cv2.saliency.StaticSaliencySpectralResidual_create()
            ok, sal_map = sal.computeSaliency(bgr)
            if ok:
                mass = sal_map.astype(np.float32)
                mass = mass / (mass.sum() + 1e-6)
                ys = np.arange(h)[:, None]
                xs = np.arange(w)[None, :]
                cy = float((ys * mass).sum())
                cx = float((xs * mass).sum())
                offset = math.sqrt((cx / w - 0.5) ** 2 + (cy / h - 0.5) ** 2)
                return cx / w, cy / h, offset
        except Exception:
            pass

    lum = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
    mass = lum / (lum.sum() + 1e-6)
    ys = np.arange(h)[:, None]
    xs = np.arange(w)[None, :]
    cy = float((ys * mass).sum())
    cx = float((xs * mass).sum())
    offset = math.sqrt((cx / w - 0.5) ** 2 + (cy / h - 0.5) ** 2)
    return cx / w, cy / h, offset


def _composition_balance(img: Image.Image) -> dict:
    cx, cy, offset = _saliency_com(img)
    # High bar = well balanced (near center); low = shifted
    value = _clip01(1.0 - offset / 0.45)
    return {
        "id": "composition_balance",
        "label": "Composition Balance",
        "value": round(value, 3),
        "raw": {"center_x": round(cx, 4), "center_y": round(cy, 4), "offset": round(offset, 4)},
        "detail": f"visual mass at ({cx:.0%}, {cy:.0%}), offset {offset:.2f} from center",
    }


def _saturation_intensity(img: Image.Image) -> dict:
    _, sat, _, _ = _hsv_arrays(img)
    mean_sat = float(sat.mean())
    value = _clip01(mean_sat / 0.65)
    return {
        "id": "saturation_intensity",
        "label": "Saturation Intensity",
        "value": round(value, 3),
        "raw": {"mean_saturation": round(mean_sat, 4)},
        "detail": f"mean saturation {mean_sat:.2f}",
    }


def _edge_complexity(img: Image.Image) -> dict:
    arr = _rgb_array(img)
    gray = (0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2] * 255).astype(
        np.uint8
    )
    if _HAS_CV2:
        edges = cv2.Canny(gray, 80, 160)
        density = float(edges.mean()) / 255.0
    else:
        gx = np.abs(np.diff(gray.astype(np.float32), axis=1)).mean()
        gy = np.abs(np.diff(gray.astype(np.float32), axis=0)).mean()
        density = _clip01((gx + gy) / 120.0)

    value = _clip01(density / 0.35)
    return {
        "id": "edge_complexity",
        "label": "Edge Complexity",
        "value": round(value, 3),
        "raw": {"edge_density": round(density, 4)},
        "detail": f"edge density {density:.3f}",
    }


def _warm_cool(img: Image.Image) -> dict:
    """Hue centroid mapped: 0=cool, 1=warm."""
    hue, sat, _, _ = _hsv_arrays(img)
    active = sat > 0.06
    if active.sum() < 10:
        centroid = 0.5
    else:
        h = hue[active]
        # Circular mean on hue circle
        ang = h * 2 * math.pi
        centroid = (math.atan2(float(np.sin(ang).mean()), float(np.cos(ang).mean())) / (2 * math.pi)) % 1.0

    # Map hue: blues/greens (~0.45–0.75) = cool, reds/oranges (~0.0–0.12, 0.88–1.0) = warm
    warm_score = 1.0 - abs(((centroid + 0.5) % 1.0) - 0.5) * 2
    warm_score = _clip01(warm_score * 0.6 + 0.2 if centroid < 0.2 or centroid > 0.85 else 0.35)
    return {
        "id": "warm_cool",
        "label": "Warm / Cool",
        "value": round(warm_score, 3),
        "raw": {"hue_centroid": round(centroid, 4)},
        "detail": f"hue centroid {centroid:.2f} ({'warm-leaning' if warm_score > 0.55 else 'cool-leaning' if warm_score < 0.45 else 'neutral'})",
    }


def compute_dissection(img: Image.Image) -> list[dict]:
    """Return ordered list of dimension dicts with value in [0, 1]."""
    img = img.convert("RGB")
    return [
        _color_harmony(img),
        _composition_balance(img),
        _saturation_intensity(img),
        _edge_complexity(img),
        _warm_cool(img),
    ]
