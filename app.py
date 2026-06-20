"""
Aesthetic Dissection Panel
Upload an image → LAION score + auditable dimension proxies + plain-English rationales.
Deploy: Hugging Face Spaces (Gradio SDK) · same file as local.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

from aesthetic_core import compute_dissection, laion_aesthetic_score
from aesthetic_core.config import DEVICE

from src.config import SHARE_LOCAL
from src.rationale import generate_rationales

CSS_PATH = Path(__file__).parent / "css" / "custom.css"
CUSTOM_CSS = CSS_PATH.read_text(encoding="utf-8") if CSS_PATH.exists() else ""


def _score_color(score: float) -> str:
    if score >= 7.0:
        return "#34c759"
    if score >= 5.5:
        return "#ff9500"
    return "#ff3b30"


def _bar_color(v: float) -> str:
    if v >= 0.72:
        return "#34c759"
    if v >= 0.45:
        return "#007aff"
    return "#8e8e93"


def _render_laion(score: float | None) -> str:
    if score is None:
        return """
        <div class="score-card empty">
          <div class="score-label">LAION Aesthetic Score</div>
          <div class="score-hint">Upload an image to analyze</div>
        </div>"""
    color = _score_color(score)
    return f"""
    <div class="score-card">
      <div class="score-label">LAION Aesthetic Score</div>
      <div class="score-value" style="color:{color}">{score:.2f}</div>
      <div class="score-sub">CLIP ViT-L/14 + sa_0_4_vit_l_14_linear.pth · 0–10</div>
      <div class="score-bar-track">
        <div class="score-bar-fill" style="width:{score/10*100:.0f}%;background:{color}"></div>
      </div>
    </div>"""


def _render_dimensions(dims: list[dict], rationales: list[str], source: str) -> str:
    if not dims:
        return '<div class="dim-empty">Dimensions will appear here after upload.</div>'

    rows = []
    for d, rationale in zip(dims, rationales):
        v = d["value"]
        pct = int(v * 100)
        color = _bar_color(v)
        detail = d.get("detail", "")
        raw = d.get("raw") or {}
        submetrics = detail or ", ".join(f"{k.replace('_', ' ')} {val}" for k, val in raw.items())
        rows.append(
            f"""
            <div class="dim-row">
              <div class="dim-head">
                <span class="dim-name">{html.escape(d['label'])}</span>
                <span class="dim-pct">{pct}%</span>
              </div>
              <div class="dim-bar-track">
                <div class="dim-bar-fill" style="width:{pct}%;background:{color}"></div>
              </div>
              <p class="dim-submetrics">{html.escape(submetrics)}</p>
              <p class="dim-rationale">{html.escape(rationale)}</p>
            </div>"""
        )

    return f"""
    <div class="dim-panel">
      <div class="dim-panel-head">
        <span>Dissection Sliders</span>
        <span class="dim-source">Rationale · {html.escape(source)}</span>
      </div>
      {''.join(rows)}
    </div>"""


def _render_json(payload: str) -> str:
    return f'<pre class="json-raw">{html.escape(payload)}</pre>'


def _render_error(message: str) -> str:
    return f"""
    <div class="score-card empty">
      <div class="score-label">Analysis failed</div>
      <div class="score-hint">{html.escape(message)}</div>
    </div>"""


def _coerce_pil_image(image) -> Image.Image | None:
    if image is None:
        return None
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, dict):
        source = image.get("path") or image.get("url")
        if not source:
            raise ValueError("Image dict missing path/url")
        return Image.open(source).convert("RGB")
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")
    return Image.fromarray(image).convert("RGB")


def analyze(image) -> tuple[str, str, str]:
    waiting = "{\n  \"status\": \"waiting for upload\"\n}"
    if image is None:
        return _render_laion(None), _render_dimensions([], [], ""), _render_json(waiting)

    try:
        pil = _coerce_pil_image(image)
        if pil is None:
            return _render_laion(None), _render_dimensions([], [], ""), _render_json(waiting)

        laion = laion_aesthetic_score(pil)
        dims = compute_dissection(pil)
        rationales, source = generate_rationales(dims, laion)

        detail = {
            "laion_aesthetic_score": laion,
            "device": str(DEVICE),
            "rationale_source": source,
            "dimensions": [{**d, "rationale": r} for d, r in zip(dims, rationales)],
        }
        return (
            _render_laion(laion),
            _render_dimensions(dims, rationales, source),
            _render_json(json.dumps(detail, indent=2)),
        )
    except Exception as exc:
        err = str(exc)
        return (
            _render_error(err),
            _render_dimensions([], [], ""),
            _render_json(json.dumps({"error": err}, indent=2)),
        )


with gr.Blocks(
    title="Aesthetic Dissection Panel",
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="gray", neutral_hue="gray"),
    css=CUSTOM_CSS,
) as demo:
    gr.HTML(
        """
        <header class="app-header">
          <p class="eyebrow">HCI · Aesthetics · XAI-light</p>
          <h1>Aesthetic Dissection</h1>
          <p class="subtitle">
            Operationalize visual quality into auditable dimensions —
            not a black-box accept/reject score.
            Sliders are <em>computable proxies</em>, not universal beauty rules;
            use them to compare and discuss, not to declare “good” or “bad.”
          </p>
        </header>
        """
    )

    with gr.Row():
        with gr.Column(scale=5):
            image_in = gr.Image(type="pil", label="Upload image", height=480)
            analyze_btn = gr.Button("Analyze", variant="primary")
            gr.Markdown("<p class='upload-hint'>Drop or click to upload · PNG / JPG</p>")
        with gr.Column(scale=6):
            laion_out = gr.HTML(_render_laion(None))
            dims_out = gr.HTML(_render_dimensions([], [], ""))

    with gr.Accordion("Raw metrics (JSON)", open=False):
        json_out = gr.HTML(_render_json('{"status": "waiting for upload"}'))

    analyze_event = dict(fn=analyze, inputs=[image_in], outputs=[laion_out, dims_out, json_out])
    image_in.change(**analyze_event)
    analyze_btn.click(**analyze_event)

    demo.queue(default_concurrency_limit=1)


if __name__ == "__main__":
    print(f"\n🎨  Aesthetic Dissection Panel · device = {DEVICE}\n")
    demo.launch(share=SHARE_LOCAL)
