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
            "high": f"Color palette shows coherent spread and contrast ({detail}) — reads curated and harmonious.",
            "moderate": f"Color harmony is acceptable but not exceptional ({detail}) — some palette tension remains.",
            "low": f"Limited hue diversity or weak contrast ({detail}) — palette may feel flat or disjointed.",
        },
        "composition_balance": {
            "high": f"Visual mass sits near center ({detail}) — balanced layout with safe text zones.",
            "moderate": f"Slight compositional drift ({detail}) — monitor crop and headline placement.",
            "low": f"Strong center-of-mass shift ({detail}) — ad text zone or focal balance may be compromised.",
        },
        "saturation_intensity": {
            "high": f"High saturation ({detail}) — pushes a bold, glamorous feel; watch for 'cheap neon' on luxury briefs.",
            "moderate": f"Moderate saturation ({detail}) — energetic but still controllable for brand tone.",
            "low": f"Muted saturation ({detail}) — restrained, premium-leaning palette.",
        },
        "edge_complexity": {
            "high": f"Rich edge structure ({detail}) — high visual detail and texture complexity.",
            "moderate": f"Moderate edge density ({detail}) — enough detail without visual noise.",
            "low": f"Low edge complexity ({detail}) — minimal, clean surfaces; may feel flat if too low.",
        },
        "warm_cool": {
            "high": f"Warm hue bias ({detail}) — inviting, skin-friendly, lifestyle-ad tone.",
            "moderate": f"Neutral temperature ({detail}) — flexible for mixed brand palettes.",
            "low": f"Cool hue bias ({detail}) — clinical, tech, or premium-minimal associations.",
        },
    }
    base = templates.get(dim_id, {}).get(lvl, f"{dim['label']} is {lvl} ({v:.0%}). {detail}")
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
