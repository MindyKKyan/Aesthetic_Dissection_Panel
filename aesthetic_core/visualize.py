"""
Diagnostic map generation — make dissection proxies visually inspectable.
"""

from __future__ import annotations

import base64
import io
import math

import numpy as np
from PIL import Image, ImageDraw

from .alignment import _palette_swatches
from .dissection import _edge_map, _hsv_arrays, _rgb_array, _saliency_com, _saliency_map

_THUMB = 280


def _resize(img: Image.Image, size: int = _THUMB) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    scale = size / max(w, h)
    if scale < 1.0:
        return img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    return img


def _to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def edge_diagnostic(img: Image.Image) -> Image.Image:
    """Cyan edges on dark background."""
    thumb = _resize(img)
    edges = _edge_map(thumb)
    h, w = edges.shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:] = (28, 28, 30)
    canvas[edges > 0] = (0, 199, 190)
    return Image.fromarray(canvas)


def saliency_diagnostic(img: Image.Image) -> Image.Image:
    """Saliency heatmap blended over thumbnail."""
    thumb = _resize(img)
    arr = np.array(thumb, dtype=np.float32)
    sal = _saliency_map(thumb)
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-6)
    # simple warm ramp: dark blue → orange
    heat = np.zeros_like(arr)
    heat[:, :, 0] = sal * 255
    heat[:, :, 1] = sal * 140
    heat[:, :, 2] = (1 - sal) * 180
    blend = (0.55 * arr + 0.45 * heat).astype(np.uint8)
    return Image.fromarray(blend)


def composition_diagnostic(img: Image.Image) -> Image.Image:
    """Rule-of-thirds grid + visual-mass centroid."""
    thumb = _resize(img)
    cx, cy, _ = _saliency_com(thumb)
    w, h = thumb.size
    overlay = thumb.copy()
    draw = ImageDraw.Draw(overlay)
    for frac in (1 / 3, 2 / 3):
        x, y = int(w * frac), int(h * frac)
        draw.line([(x, 0), (x, h)], fill=(255, 255, 255, 180), width=1)
        draw.line([(0, y), (w, y)], fill=(255, 255, 255, 180), width=1)
    px, py = int(cx * w), int(cy * h)
    r = max(6, min(w, h) // 28)
    draw.ellipse([(px - r, py - r), (px + r, py + r)], outline=(255, 59, 48), width=3)
    draw.line([(px - r - 4, py), (px + r + 4, py)], fill=(255, 59, 48), width=2)
    draw.line([(px, py - r - 4), (px, py + r + 4)], fill=(255, 59, 48), width=2)
    return overlay


def palette_diagnostic(img: Image.Image) -> Image.Image:
    """Horizontal strip of dominant palette swatches."""
    swatches = _palette_swatches(img, k=6)
    strip_h = 72
    strip_w = _THUMB
    block = strip_w // len(swatches)
    canvas = Image.new("RGB", (strip_w, strip_h), (245, 245, 247))
    draw = ImageDraw.Draw(canvas)
    for i, hex_color in enumerate(swatches):
        x0 = i * block
        draw.rectangle([(x0, 8), (x0 + block - 4, strip_h - 8)], fill=hex_color)
    return canvas


def _hsv01_to_rgb(h: float, s: float, v: float) -> tuple[int, int, int]:
    import colorsys

    r, g, b = colorsys.hsv_to_rgb(h % 1.0, float(np.clip(s, 0, 1)), float(np.clip(v, 0, 1)))
    return int(r * 255), int(g * 255), int(b * 255)


def _hsv01_to_rgb_np(h: np.ndarray, s: float, v: float) -> np.ndarray:
    h = np.asarray(h, dtype=np.float64) % 1.0
    s, v = float(s), float(v)
    i = np.floor(h * 6).astype(int) % 6
    f = h * 6 - np.floor(h * 6)
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=-1)


def hue_wheel_diagnostic(img: Image.Image) -> Image.Image:
    """
    Full-spectrum ring + polar hue histogram + warm/cool gradient bar.
    """
    thumb = _resize(img)
    hue, sat, val, _ = _hsv_arrays(thumb)
    size = _THUMB
    cx = cy = size // 2
    canvas = np.full((size, size, 3), 245 / 255.0, dtype=np.float32)

    yy, xx = np.ogrid[:size, :size]
    dx = xx.astype(np.float32) - cx
    dy = yy.astype(np.float32) - cy
    radius = np.sqrt(dx * dx + dy * dy)
    angle = np.arctan2(dx, -dy) % (2 * math.pi)
    hue_at = angle / (2 * math.pi)

    outer_r = size // 2 - 12
    inner_r = outer_r - 34
    ring = (radius >= inner_r) & (radius <= outer_r)
    canvas[ring] = _hsv01_to_rgb_np(hue_at[ring], 0.88, 0.96)

    centroid_h = None
    active = sat > 0.06
    if active.sum() > 5:
        h_active = hue[active]
        weights = (sat[active] * val[active]).astype(np.float32)
        bins = 36
        hist, _ = np.histogram(h_active, bins=bins, range=(0, 1), weights=weights)
        hist = hist.astype(np.float32)
        if hist.max() > 0:
            hist /= hist.max()

        max_bar = inner_r - 14
        for i, strength in enumerate(hist):
            if strength < 0.07:
                continue
            a0 = 2 * math.pi * i / bins
            a1 = 2 * math.pi * (i + 1) / bins
            in_bin = (angle >= a0) & (angle < a1) & (radius <= max(10, max_bar * strength))
            bar_rgb = np.array(_hsv01_to_rgb((i + 0.5) / bins, 0.7, 0.82), dtype=np.float32) / 255.0
            canvas[in_bin] = canvas[in_bin] * 0.25 + bar_rgb * 0.75

        ang = h_active * 2 * math.pi
        centroid = math.atan2(float(np.sin(ang).mean()), float(np.cos(ang).mean()))
        centroid_h = (centroid % (2 * math.pi)) / (2 * math.pi)

    out = Image.fromarray(np.clip(canvas * 255, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(out)

    if centroid_h is not None:
        mark_a = centroid_h * 2 * math.pi
        mx = cx + int((outer_r - 18) * math.sin(mark_a))
        my = cy - int((outer_r - 18) * math.cos(mark_a))
        dot = _hsv01_to_rgb(centroid_h, 0.85, 0.92)
        draw.line([(cx, cy), (mx, my)], fill=(60, 60, 67), width=2)
        draw.ellipse([(mx - 8, my - 8), (mx + 8, my + 8)], fill=dot, outline=(255, 255, 255), width=2)

    bar_y0, bar_y1 = size - 28, size - 14
    bar_x0, bar_x1 = 24, size - 24
    steps = bar_x1 - bar_x0
    for i in range(steps):
        t = i / max(steps - 1, 1)
        h = 0.58 * (1 - t) + 0.02 * t
        c = _hsv01_to_rgb(h, 0.75, 0.9)
        draw.line([(bar_x0 + i, bar_y0), (bar_x0 + i, bar_y1)], fill=c, width=1)
    if centroid_h is not None:
        warmness = 1.0 - min(1.0, abs(((centroid_h - 0.58 + 0.5) % 1.0) - 0.5) * 2)
        tx = int(bar_x0 + warmness * (bar_x1 - bar_x0))
        draw.polygon([(tx, bar_y0 - 2), (tx - 6, bar_y0 - 10), (tx + 6, bar_y0 - 10)], fill=(29, 29, 31))
    draw.text((bar_x0, size - 12), "Cool", fill=(70, 110, 190))
    draw.text((bar_x1 - 34, size - 12), "Warm", fill=(210, 85, 45))

    return out


def build_diagnostic_bundle(img: Image.Image) -> dict[str, str]:
    """Return data URLs keyed by diagnostic id."""
    img = img.convert("RGB")
    return {
        "edges": _to_data_url(edge_diagnostic(img)),
        "saliency": _to_data_url(saliency_diagnostic(img)),
        "composition": _to_data_url(composition_diagnostic(img)),
        "palette": _to_data_url(palette_diagnostic(img)),
        "hue_wheel": _to_data_url(hue_wheel_diagnostic(img)),
    }
