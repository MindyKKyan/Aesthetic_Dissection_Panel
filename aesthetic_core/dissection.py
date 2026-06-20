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


def _level(v: float) -> str:
    if v >= 0.72:
        return "high"
    if v >= 0.45:
        return "moderate"
    return "low"


def _pct(v: float) -> str:
    if v < 0.005:
        return "0%"
    if v < 0.1:
        return f"{v * 100:.1f}%"
    return f"{int(round(v * 100))}%"


def _edge_map(img: Image.Image) -> np.ndarray:
    """Binary edge map used by texture metric and diagnostics."""
    arr = _rgb_array(img)
    lum = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
    gray = (lum * 255).astype(np.uint8)
    if _HAS_CV2:
        h, w = gray.shape
        scale = min(1.0, 512 / max(h, w))
        if scale < 1.0:
            gray = cv2.resize(
                gray,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )
        median = float(np.median(gray))
        lower = int(max(0, 0.66 * median))
        upper = int(min(255, 1.33 * median))
        if upper - lower < 20:
            lower, upper = 50, 150
        return cv2.Canny(gray, lower, upper)
    gx = np.abs(np.diff(lum, axis=1)).mean(axis=0, keepdims=True)
    gy = np.abs(np.diff(lum, axis=0))
    edge = np.zeros_like(lum, dtype=np.uint8)
    edge[:, :-1] = np.maximum(edge[:, :-1], (gx * 400).astype(np.uint8))
    edge[:-1, :] = np.maximum(edge[:-1, :], (gy * 400).astype(np.uint8))
    return edge


def _saliency_map(img: Image.Image) -> np.ndarray:
    """Saliency map in [0, 1], fallback to luminance."""
    arr = _rgb_array(img)
    h, w, _ = arr.shape
    if _HAS_CV2:
        bgr = (arr * 255).astype(np.uint8)[:, :, ::-1]
        try:
            sal = cv2.saliency.StaticSaliencySpectralResidual_create()
            ok, sal_map = sal.computeSaliency(bgr)
            if ok:
                return sal_map.astype(np.float32)
        except Exception:
            pass
    lum = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
    return lum.astype(np.float32)


def _saliency_com(img: Image.Image) -> tuple[float, float, float]:
    """Return (cx, cy, offset) in 0–1 coords; offset = distance from center."""
    arr = _rgb_array(img)
    h, w, _ = arr.shape
    mass = _saliency_map(img)
    mass = mass / (mass.sum() + 1e-6)
    ys = np.arange(h)[:, None]
    xs = np.arange(w)[None, :]
    cy = float((ys * mass).sum())
    cx = float((xs * mass).sum())
    offset = math.sqrt((cx / w - 0.5) ** 2 + (cy / h - 0.5) ** 2)
    return cx / w, cy / h, offset


def composition_offset(img: Image.Image) -> float:
    """Saliency center-of-mass offset from image center (0 = centered, higher = shifted)."""
    _, _, offset = _saliency_com(img.convert("RGB"))
    return round(offset, 4)


def _color_harmony(img: Image.Image) -> dict:
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
    lvl = _level(value)
    interpretations = {
        "high": "wide hue spread with strong light–dark separation",
        "moderate": "balanced palette structure — neither flat nor chaotic",
        "low": "narrow palette or low luminance contrast",
    }
    return {
        "id": "color_harmony",
        "label": "Color Structure",
        "value": round(value, 3),
        "level": lvl,
        "interpretation": interpretations[lvl],
        "sub_metrics": [
            {"label": "Hue diversity", "value": round(spread, 3), "pct": _pct(spread)},
            {"label": "Luminance contrast", "value": round(contrast, 3), "pct": _pct(contrast)},
        ],
        "raw": {
            "hue_entropy_norm": round(spread, 4),
            "luminance_contrast": round(contrast, 4),
        },
        "detail": f"hue diversity {spread:.2f}, luminance contrast {contrast:.2f}",
    }


def _composition_balance(img: Image.Image) -> dict:
    cx, cy, offset = _saliency_com(img)
    centeredness = _clip01(1.0 - offset / 0.45)
    value = centeredness
    lvl = _level(value)
    interpretations = {
        "high": "visual mass sits near the center — symmetric, stable layout",
        "moderate": "mild asymmetry — subject or mass slightly off-center",
        "low": "strong asymmetry — dynamic or edge-weighted composition",
    }
    return {
        "id": "composition_balance",
        "label": "Visual Mass / Composition",
        "value": round(value, 3),
        "level": lvl,
        "interpretation": interpretations[lvl],
        "sub_metrics": [
            {"label": "Centeredness", "value": round(centeredness, 3), "pct": _pct(centeredness)},
            {"label": "Offset from center", "value": round(offset, 3), "pct": _pct(offset)},
        ],
        "raw": {"center_x": round(cx, 4), "center_y": round(cy, 4), "offset": round(offset, 4)},
        "detail": f"visual mass at ({cx:.0%}, {cy:.0%}), offset {offset:.2f} from center",
    }


def _saturation_intensity(img: Image.Image) -> dict:
    _, sat, _, _ = _hsv_arrays(img)
    mean_sat = float(sat.mean())
    value = _clip01(mean_sat / 0.65)
    lvl = _level(value)
    if mean_sat < 0.02:
        lvl = "low"
    interpretations = {
        "high": "vivid, saturated colors dominate the image",
        "moderate": "moderate chroma — neither muted nor neon",
        "low": "muted or achromatic palette — grayscale or pastel",
    }
    return {
        "id": "saturation_intensity",
        "label": "Color Vividness",
        "value": round(value, 3),
        "level": lvl,
        "interpretation": interpretations[lvl],
        "sub_metrics": [
            {"label": "Mean saturation", "value": round(mean_sat, 3), "pct": _pct(mean_sat)},
        ],
        "raw": {"mean_saturation": round(mean_sat, 4)},
        "detail": f"mean saturation {mean_sat:.2f}",
    }


def _edge_complexity(img: Image.Image) -> dict:
    edges = _edge_map(img)
    density = float(edges.mean()) / 255.0
    value = _clip01(density / 0.12)
    lvl = _level(value)
    interpretations = {
        "high": "rich fine edges — textured, detailed surfaces",
        "moderate": "visible structure without visual noise",
        "low": "smooth, minimal-detail surfaces (not necessarily bad)",
    }
    return {
        "id": "edge_complexity",
        "label": "Visual Texture / Detail",
        "value": round(value, 3),
        "level": lvl,
        "interpretation": interpretations[lvl],
        "sub_metrics": [
            {"label": "Edge pixel density", "value": round(density, 3), "pct": _pct(density)},
        ],
        "raw": {"edge_density": round(density, 4)},
        "detail": f"edge density {density:.3f}",
    }


def _warm_cool(img: Image.Image) -> dict:
    hue, sat, _, _ = _hsv_arrays(img)
    active = sat > 0.06
    if active.sum() < 10:
        centroid = 0.5
    else:
        h = hue[active]
        ang = h * 2 * math.pi
        centroid = (math.atan2(float(np.sin(ang).mean()), float(np.cos(ang).mean())) / (2 * math.pi)) % 1.0

    warm_score = 1.0 - abs(((centroid + 0.5) % 1.0) - 0.5) * 2
    warm_score = _clip01(warm_score * 0.6 + 0.2 if centroid < 0.2 or centroid > 0.85 else 0.35)
    lvl = _level(warm_score)
    temp_label = "warm-leaning" if warm_score > 0.55 else "cool-leaning" if warm_score < 0.45 else "neutral"
    interpretations = {
        "high": "warm hue bias — reds, oranges, yellows dominate",
        "moderate": "neutral temperature — mixed or balanced hues",
        "low": "cool hue bias — blues, greens, cyans dominate",
    }
    return {
        "id": "warm_cool",
        "label": "Color Temperature",
        "value": round(warm_score, 3),
        "level": lvl,
        "interpretation": interpretations[lvl],
        "sub_metrics": [
            {"label": "Warmth index", "value": round(warm_score, 3), "pct": _pct(warm_score)},
            {"label": "Hue centroid", "value": round(centroid, 3), "pct": f"{centroid:.2f}"},
        ],
        "raw": {"hue_centroid": round(centroid, 4)},
        "detail": f"hue centroid {centroid:.2f} ({temp_label})",
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
