"""
Plain-English rationales — OpenAI / HF Inference / template fallback.
"""

from __future__ import annotations

import json
import urllib.request

from .config import HF_TOKEN, OPENAI_API_KEY


def _level(v: float) -> str:
    if v >= 0.72:
        return "high"
    if v >= 0.45:
        return "moderate"
    return "low"


def _template_rationale(dim: dict, laion_score: float) -> str:
    v = dim["value"]
    lvl = _level(v)
    dim_id = dim["id"]
    detail = dim.get("detail", "")

    templates = {
        "color_harmony": {
            "high": f"Wide hue spread with strong light–dark separation ({detail}) — reads as a structured palette.",
            "moderate": f"Balanced color structure ({detail}) — neither flat nor chaotic.",
            "low": f"Narrow palette or low contrast ({detail}) — may read as muted or monochrome.",
        },
        "composition_balance": {
            "high": f"Visual mass near center ({detail}) — symmetric, stable framing.",
            "moderate": f"Mild asymmetry ({detail}) — subject slightly off-center; can feel dynamic.",
            "low": f"Strong asymmetry ({detail}) — edge-weighted or rule-breaking layout (not inherently bad).",
        },
        "saturation_intensity": {
            "high": f"Vivid chroma ({detail}) — bold, energetic color presence.",
            "moderate": f"Moderate vividness ({detail}) — controlled color energy.",
            "low": f"Muted or achromatic ({detail}) — restrained, minimal, or grayscale palette.",
        },
        "edge_complexity": {
            "high": f"Rich edge structure ({detail}) — textured, detailed surfaces.",
            "moderate": f"Moderate texture ({detail}) — visible structure without noise.",
            "low": f"Smooth surfaces ({detail}) — minimal fine detail; clean or flat depending on intent.",
        },
        "warm_cool": {
            "high": f"Warm hue bias ({detail}) — reds/oranges/yellows dominate the mood.",
            "moderate": f"Neutral temperature ({detail}) — mixed hues, flexible mood.",
            "low": f"Cool hue bias ({detail}) — blues/greens dominate the mood.",
        },
    }
    interpretation = dim.get("interpretation", "")
    base = templates.get(dim_id, {}).get(lvl, f"{dim['label']} is {lvl} ({v:.0%}). {detail}")
    if interpretation and interpretation not in base:
        base = f"{interpretation.capitalize()}. {base}"
    if laion_score >= 6.5 and v >= 0.7:
        return base
    if laion_score < 5.0 and v < 0.4:
        return base + " Aligns with the lower LAION aesthetic score."
    return base


def _openai_rationales(dims: list[dict], laion_score: float) -> list[str] | None:
    if not OPENAI_API_KEY:
        return None
    try:
        from openai import OpenAI

        client = OpenAI(api_key=OPENAI_API_KEY)
        payload = {
            "laion_score": laion_score,
            "dimensions": [
                {"label": d["label"], "value_0_1": d["value"], "raw": d.get("raw"), "detail": d.get("detail")}
                for d in dims
            ],
        }
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an art director writing one-sentence plain-English rationales for aesthetic "
                        "diagnostics. Be specific, cite direction (+/-). Return JSON: {\"rationales\": [str×N]} "
                        "same order as input dimensions."
                    ),
                },
                {"role": "user", "content": json.dumps(payload)},
            ],
            response_format={"type": "json_object"},
            max_tokens=600,
            temperature=0.4,
        )
        data = json.loads(resp.choices[0].message.content)
        lines = data.get("rationales", [])
        if len(lines) == len(dims):
            return lines
    except Exception:
        pass
    return None


def _hf_rationales(dims: list[dict], laion_score: float) -> list[str] | None:
    if not HF_TOKEN:
        return None
    try:
        import urllib.parse

        prompt = (
            f"LAION aesthetic score: {laion_score}/10. "
            f"For each dimension write ONE short plain-English rationale sentence:\n"
            + "\n".join(f"- {d['label']}: {d['value']:.2f} ({d.get('detail')})" for d in dims)
            + "\nReturn only numbered lines 1-5."
        )
        body = json.dumps({"inputs": prompt, "parameters": {"max_new_tokens": 400}}).encode()
        req = urllib.request.Request(
            "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta",
            data=body,
            headers={"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            out = json.loads(resp.read().decode())
        text = out[0]["generated_text"] if isinstance(out, list) else out.get("generated_text", "")
        lines = [ln.strip().lstrip("0123456789.) ") for ln in text.split("\n") if ln.strip()]
        if len(lines) >= len(dims):
            return lines[: len(dims)]
    except Exception:
        pass
    return None


def generate_rationales(dims: list[dict], laion_score: float) -> tuple[list[str], str]:
    """Return (rationales, source_label)."""
    for fn, label in [
        (_openai_rationales, "GPT-4o mini"),
        (_hf_rationales, "HF Inference"),
    ]:
        result = fn(dims, laion_score)
        if result:
            return result, label
    return [_template_rationale(d, laion_score) for d in dims], "rule-based template"
