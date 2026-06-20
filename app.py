"""
Aesthetic Dissection Panel
Upload an image → LAION score + auditable dimension proxies + plain-English rationales.
Deploy: Hugging Face Spaces (Gradio SDK) · same file as local.
"""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
from PIL import Image

from src.config import DEVICE, SHARE_LOCAL
from src.dissection import compute_dissection
from src.laion_scorer import laion_aesthetic_score
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
        rows.append(
            f"""
            <div class="dim-row">
              <div class="dim-head">
                <span class="dim-name">{d['label']}</span>
                <span class="dim-pct">{pct}%</span>
              </div>
              <div class="dim-bar-track">
                <div class="dim-bar-fill" style="width:{pct}%;background:{color}"></div>
              </div>
              <p class="dim-rationale">{rationale}</p>
            </div>"""
        )

    return f"""
    <div class="dim-panel">
      <div class="dim-panel-head">
        <span>Dissection Sliders</span>
        <span class="dim-source">Rationale · {source}</span>
      </div>
      {''.join(rows)}
    </div>"""


def analyze(image) -> tuple[str, str, str]:
    if image is None:
        return _render_laion(None), _render_dimensions([], [], ""), "{}"

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    laion = laion_aesthetic_score(image)
    dims = compute_dissection(image)
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
        json.dumps(detail, indent=2),
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
          </p>
        </header>
        """
    )

    with gr.Row(equal_height=True):
        with gr.Column(scale=5):
            image_in = gr.Image(
                type="pil",
                label=None,
                show_label=False,
                height=480,
                sources=["upload", "clipboard"],
            )
            gr.Markdown(
                "<p class='upload-hint'>Drop or click to upload · PNG / JPG</p>",
                elem_classes=["upload-hint-md"],
            )
        with gr.Column(scale=6):
            laion_out = gr.HTML(_render_laion(None))
            dims_out = gr.HTML(_render_dimensions([], [], ""))

    with gr.Accordion("Raw metrics (JSON)", open=False):
        json_out = gr.Code(language="json", label=None, show_label=False, lines=12)

    image_in.change(
        fn=analyze,
        inputs=[image_in],
        outputs=[laion_out, dims_out, json_out],
    )


if __name__ == "__main__":
    print(f"\n🎨  Aesthetic Dissection Panel · device = {DEVICE}\n")
    demo.launch(share=SHARE_LOCAL)
