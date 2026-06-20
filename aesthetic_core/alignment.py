"""
Style-alignment metrics — CLIP similarity, LAION delta, palette distance, composition.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
from PIL import Image

from .dissection import _hsv_arrays, composition_offset
from .laion_scorer import clip_embedding_768, laion_aesthetic_score

STYLE_LABELS = [
    "commercial advertisement",
    "fine art photography",
    "minimalist design",
    "luxury brand aesthetic",
    "warm cozy mood",
    "cool clinical mood",
    "vibrant saturated colors",
    "muted pastel palette",
    "editorial fashion",
    "natural landscape",
    "urban street style",
    "vintage retro look",
]


def _rgb_to_hex(r: float, g: float, b: float) -> str:
    return "#{:02x}{:02x}{:02x}".format(
        int(max(0, min(255, r * 255))),
        int(max(0, min(255, g * 255))),
        int(max(0, min(255, b * 255))),
    )


def _palette_swatches(img: Image.Image, k: int = 5) -> list[str]:
    """Dominant colors as hex swatches via coarse HSV binning."""
    hue, sat, val, arr = _hsv_arrays(img)
    active = sat > 0.1
    if active.sum() < 20:
        mean = arr.mean(axis=(0, 1))
        return [_rgb_to_hex(*mean)]

    h_active = hue[active]
    s_active = sat[active]
    v_active = val[active]
    bins = np.stack(
        [
            (h_active * 12).astype(int) % 12,
            (s_active * 4).astype(int),
            (v_active * 4).astype(int),
        ],
        axis=1,
    )
    keys = [tuple(row) for row in bins]
    counts: dict[tuple, int] = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1
    top_keys = sorted(counts, key=counts.get, reverse=True)[:k]

    swatches = []
    for key in top_keys:
        mask = np.all(bins == np.array(key), axis=1)
        if mask.sum() == 0:
            continue
        mean_rgb = np.array(
            [
                arr[:, :, 0][active][mask].mean(),
                arr[:, :, 1][active][mask].mean(),
                arr[:, :, 2][active][mask].mean(),
            ]
        )
        swatches.append(_rgb_to_hex(*mean_rgb))
    return swatches or [_rgb_to_hex(*arr.mean(axis=(0, 1)))]


def _hsv_histogram(img: Image.Image, bins: int = 36) -> np.ndarray:
    hue, sat, val, _ = _hsv_arrays(img)
    active = sat > 0.08
    if active.sum() < 10:
        return np.ones(bins) / bins
    hist, _ = np.histogram(hue[active], bins=bins, range=(0, 1), density=True)
    hist = hist + 1e-8
    return hist / hist.sum()


@torch.no_grad()
def anchor_embedding(anchors: Sequence[Image.Image]) -> torch.Tensor:
    """Average normalized CLIP embeddings from 1–5 anchor images."""
    if not anchors:
        raise ValueError("At least one anchor image is required")
    embs = [clip_embedding_768(a.convert("RGB")) for a in anchors]
    stacked = torch.cat(embs, dim=0)
    mean = stacked.mean(dim=0, keepdim=True)
    return mean / mean.norm(dim=-1, keepdim=True)


@torch.no_grad()
def clip_cosine_sim(anchor_emb: torch.Tensor, img: Image.Image) -> float:
    """Semantic closeness in [0, 1] (cosine mapped from [-1, 1])."""
    gen_emb = clip_embedding_768(img.convert("RGB"))
    cos = float((anchor_emb * gen_emb).sum().item())
    return round((cos + 1.0) / 2.0, 4)


@torch.no_grad()
def _clip_zero_shot_labels(img: Image.Image, labels: Sequence[str], top_k: int = 5) -> list[dict]:
    from .config import DEVICE
    from .laion_scorer import _load_clip

    model, processor = _load_clip()
    inp_img = processor(images=img.convert("RGB"), return_tensors="pt")
    pixel_values = inp_img["pixel_values"].to(DEVICE)
    vision_out = model.vision_model(pixel_values=pixel_values)
    cls = vision_out.last_hidden_state[:, 0, :]
    img_feat = model.visual_projection(cls)
    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

    text_inp = processor(text=list(labels), return_tensors="pt", padding=True)
    text_out = model.text_model(input_ids=text_inp["input_ids"].to(DEVICE))
    txt_feat = model.text_projection(text_out.pooler_output)
    txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

    logits = (img_feat @ txt_feat.T).squeeze(0)
    probs = torch.softmax(logits * 100, dim=0)
    top_idx = probs.argsort(descending=True)[:top_k]
    return [
        {"label": labels[int(i)], "score": round(float(probs[i].item()), 4)}
        for i in top_idx
    ]


def delta_aesthetic(anchor: Image.Image, img: Image.Image) -> dict:
    """LAION score delta with direction and interpretive hint."""
    anchor_score = laion_aesthetic_score(anchor)
    gen_score = laion_aesthetic_score(img)
    delta = round(gen_score - anchor_score, 2)
    if delta > 0.4:
        direction = "up"
        hint = "more globally polished / fine-art leaning"
    elif delta < -0.4:
        direction = "down"
        hint = "more commercial / less gallery-polished"
    else:
        direction = "neutral"
        hint = "similar aesthetic register to anchor"
    return {
        "anchor_score": anchor_score,
        "generated_score": gen_score,
        "delta": delta,
        "direction": direction,
        "hint": hint,
    }


def palette_distance(anchor: Image.Image, img: Image.Image) -> dict:
    """HSV hue histogram L2 distance + palette swatches."""
    h_anchor = _hsv_histogram(anchor)
    h_gen = _hsv_histogram(img)
    dist = float(np.linalg.norm(h_anchor - h_gen))
    max_dist = math.sqrt(2.0)
    closeness = round(max(0.0, 1.0 - dist / max_dist), 4)
    anchor_swatches = _palette_swatches(anchor)
    gen_swatches = _palette_swatches(img)
    return {
        "histogram_l2": round(dist, 4),
        "closeness": closeness,
        "anchor_palette": anchor_swatches,
        "generated_palette": gen_swatches,
        "mood": _palette_mood(anchor, img),
    }


def _palette_mood(anchor: Image.Image, img: Image.Image) -> str:
    _, sat_a, val_a, _ = _hsv_arrays(anchor)
    _, sat_g, val_g, _ = _hsv_arrays(img)
    warm_a = _warm_cool_label(anchor)
    warm_g = _warm_cool_label(img)
    sat_delta = float(sat_g.mean() - sat_a.mean())
    parts = []
    if warm_a != warm_g:
        parts.append(f"temperature shift ({warm_a} → {warm_g})")
    if abs(sat_delta) > 0.08:
        parts.append("higher saturation" if sat_delta > 0 else "lower saturation")
    if abs(float(val_g.mean() - val_a.mean())) > 0.1:
        parts.append("brighter" if val_g.mean() > val_a.mean() else "darker")
    return ", ".join(parts) if parts else "similar color mood"


def _warm_cool_label(img: Image.Image) -> str:
    hue, sat, _, _ = _hsv_arrays(img)
    active = sat > 0.06
    if active.sum() < 10:
        return "neutral"
    h = hue[active]
    ang = h * 2 * math.pi
    centroid = (math.atan2(float(np.sin(ang).mean()), float(np.cos(ang).mean())) / (2 * math.pi)) % 1.0
    if centroid < 0.15 or centroid > 0.85:
        return "warm"
    if 0.45 < centroid < 0.65:
        return "cool"
    return "neutral"


def composition_offset_delta(anchor: Image.Image, img: Image.Image) -> dict:
    """Saliency center-of-mass offset difference."""
    off_a = composition_offset(anchor)
    off_g = composition_offset(img)
    delta = round(abs(off_g - off_a), 4)
    closeness = round(max(0.0, 1.0 - delta / 0.35), 4)
    return {
        "anchor_offset": off_a,
        "generated_offset": off_g,
        "delta": delta,
        "closeness": closeness,
    }


def alignment_report(
    anchors: Sequence[Image.Image],
    generated: Image.Image,
    anchor_emb: torch.Tensor | None = None,
) -> dict:
    """Full alignment report for one generated image vs anchor set."""
    primary = anchors[0].convert("RGB")
    gen = generated.convert("RGB")
    emb = anchor_emb if anchor_emb is not None else anchor_embedding(anchors)

    clip_sim = clip_cosine_sim(emb, gen)
    aesthetic = delta_aesthetic(primary, gen)
    palette = palette_distance(primary, gen)
    composition = composition_offset_delta(primary, gen)
    labels = _clip_zero_shot_labels(gen, STYLE_LABELS, top_k=5)

    overall = round(
        0.4 * clip_sim
        + 0.2 * max(0.0, 1.0 - abs(aesthetic["delta"]) / 3.0)
        + 0.2 * palette["closeness"]
        + 0.2 * composition["closeness"],
        4,
    )

    return {
        "clip_similarity": clip_sim,
        "delta_aesthetic": aesthetic,
        "palette": palette,
        "composition": composition,
        "top_labels": labels,
        "overall_alignment": overall,
    }
