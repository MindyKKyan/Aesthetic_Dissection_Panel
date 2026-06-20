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
from aesthetic_core.depth_layers import separate_depth_layers
from aesthetic_core.visualize import build_diagnostic_bundle

from src.config import SHARE_LOCAL
from src.rationale import generate_rationales

CSS_PATH = Path(__file__).parent / "css" / "custom.css"
CUSTOM_CSS = CSS_PATH.read_text(encoding="utf-8") if CSS_PATH.exists() else ""

LANGS = {
    "en": {
        "eyebrow": "HCI · Aesthetics · XAI-light",
        "title": "Aesthetic Dissection",
        "subtitle": "Operationalize visual quality into auditable dimensions, instead of a black-box accept/reject score.",
        "language_label": "Language / 语言",
        "upload_label": "Upload image",
        "analyze_btn": "Analyze",
        "upload_hint": "Drop or click to upload · PNG / JPG",
        "guide_title": "Before you start",
        "guide_body": "Upload one image, then click Analyze. The app estimates a LAION aesthetic score, interpretable visual proxies, hue/temperature, and foreground-to-background depth layers.",
        "guide_steps": [
            "Use a full image instead of a screenshot crop when possible.",
            "Bars are descriptive signals, not universal beauty judgments.",
            "Depth is relative monocular estimation, not metric 3D reconstruction.",
        ],
        "score_label": "LAION Aesthetic Score",
        "score_hint": "Upload an image to analyze",
        "score_sub": "CLIP ViT-L/14 + sa_0_4_vit_l_14_linear.pth · 0-10",
        "dimensions_empty": "Dimensions will appear here after upload.",
        "dimensions_title": "Dissection Proxies",
        "dimensions_source_prefix": "descriptive · not prescriptive · ",
        "dimensions_disclaimer": "Bars measure what is present, not whether the image is \"good.\" Pair with LAION and your own taste.",
        "hue_title": "Hue & temperature",
        "hue_hint": "Upload to see color ring and warm/cool bar",
        "hue_caption": "Color ring · hue histogram · warm/cool bar",
        "proxy_empty": "Proxy maps will appear after upload.",
        "proxy_title": "Proxy diagnostics",
        "proxy_desc": "Edge, saliency, composition centroid, and palette maps tied to the dissection bars.",
        "cards": {
            "edges": ("Edge / texture map", "Contour and fine structure"),
            "saliency": ("Attention heatmap", "Where the eye is likely drawn"),
            "composition": ("Composition grid", "Rule-of-thirds + visual mass"),
            "palette": ("Dominant palette", "Coarse color clusters"),
        },
        "depth_unavailable": "Depth separation unavailable: ",
        "depth_empty": "Depth layers will appear after upload.",
        "depth_title": "Depth layer separation",
        "depth_desc": "Foreground / midground / background split from monocular depth.",
        "depth_note_prefix": "Monocular depth via ",
        "depth_note_suffix": " — relative near/far split (not metric 3D).",
        "near": "Near",
        "mid": "Mid",
        "far": "Far",
        "spread": "Depth spread",
        "depth_cards": {
            "depth_map": ("Depth map", "Brighter = nearer to camera"),
            "foreground": ("Foreground (near)", "Closest depth band"),
            "midground": ("Midground", "Middle depth band"),
            "background": ("Background (far)", "Farthest depth band"),
        },
        "visual_empty": "Upload an image to see visual analysis maps.",
        "visual_label": "Visual analysis — proxies, composition, edges & depth layers",
        "json_label": "Raw metrics (JSON)",
        "json_copy_btn": "Copy JSON",
        "json_copied_btn": "Copied!",
        "waiting_json": '{\n  "status": "waiting for upload"\n}',
        "note": "Proxy bars are descriptive signals, not universal beauty judgments.",
        "analysis_failed": "Analysis failed",
    },
    "zh-Hans": {
        "eyebrow": "HCI · 美学 · 轻量可解释性",
        "title": "审美拆解面板",
        "subtitle": "把视觉质量拆成可审计的维度，而不是只给一个黑箱式的好/坏分数。",
        "language_label": "语言 / Language",
        "upload_label": "上传图片",
        "analyze_btn": "开始分析",
        "upload_hint": "拖拽或点击上传 · PNG / JPG",
        "guide_title": "使用说明",
        "guide_body": "上传一张图片后点击“开始分析”。系统会输出 LAION 审美分数、可解释的视觉代理指标、色相/冷暖分布，以及前中后景深度分层。",
        "guide_steps": [
            "尽量上传完整图片，而不是局部截图。",
            "这些条形指标是描述性信号，不是绝对审美判断。",
            "深度结果是相对单目估计，不是真实 3D 距离。",
        ],
        "score_label": "LAION 审美分数",
        "score_hint": "上传图片后开始分析",
        "score_sub": "CLIP ViT-L/14 + sa_0_4_vit_l_14_linear.pth · 0-10",
        "dimensions_empty": "上传后，这里会显示拆解维度。",
        "dimensions_title": "拆解代理指标",
        "dimensions_source_prefix": "描述性 · 非规定性 · ",
        "dimensions_disclaimer": "这些条形图衡量的是“图中有什么”，不是“图是否好看”。请结合 LAION 分数和你自己的审美判断。",
        "hue_title": "色相与冷暖",
        "hue_hint": "上传后可查看色环与冷暖条",
        "hue_caption": "色环 · 色相直方图 · 冷暖条",
        "proxy_empty": "上传后，这里会显示代理可视化图。",
        "proxy_title": "代理诊断图",
        "proxy_desc": "边缘、显著性、构图质心与色板图，对应上方的拆解条形指标。",
        "cards": {
            "edges": ("边缘 / 纹理图", "轮廓与细节结构"),
            "saliency": ("注意力热力图", "视觉更容易被吸引的位置"),
            "composition": ("构图网格", "三分法 + 视觉重心"),
            "palette": ("主色板", "粗粒度颜色簇"),
        },
        "depth_unavailable": "深度分层暂不可用：",
        "depth_empty": "上传后，这里会显示前中后景深度分层。",
        "depth_title": "深度图层分离",
        "depth_desc": "根据单目深度估计，将图像拆成前景 / 中景 / 背景。",
        "depth_note_prefix": "单目深度模型 ",
        "depth_note_suffix": " · 输出相对远近关系，不是真实 3D 距离。",
        "near": "前景",
        "mid": "中景",
        "far": "背景",
        "spread": "深度离散度",
        "depth_cards": {
            "depth_map": ("深度图", "越亮越靠近镜头"),
            "foreground": ("前景（近）", "最近的深度带"),
            "midground": ("中景", "中间深度带"),
            "background": ("背景（远）", "最远的深度带"),
        },
        "visual_empty": "上传图片后可查看可视化分析结果。",
        "visual_label": "可视化分析——代理图、构图、边缘与深度图层",
        "json_label": "原始指标（JSON）",
        "json_copy_btn": "复制 JSON",
        "json_copied_btn": "已复制",
        "waiting_json": '{\n  "status": "等待上传图片"\n}',
        "note": "这些代理条形指标是描述性信号，不是绝对审美判断。",
        "analysis_failed": "分析失败",
    },
    "zh-Hant": {
        "eyebrow": "HCI · 美學 · 輕量可解釋性",
        "title": "審美拆解面板",
        "subtitle": "把視覺品質拆成可審計的維度，而不是只給一個黑箱式的好/壞分數。",
        "language_label": "語言 / Language",
        "upload_label": "上傳圖片",
        "analyze_btn": "開始分析",
        "upload_hint": "拖曳或點擊上傳 · PNG / JPG",
        "guide_title": "使用說明",
        "guide_body": "上傳一張圖片後點擊「開始分析」。系統會輸出 LAION 審美分數、可解釋的視覺代理指標、色相/冷暖分佈，以及前中後景深度分層。",
        "guide_steps": [
            "盡量上傳完整圖片，而不是局部截圖。",
            "這些條形指標是描述性訊號，不是絕對審美判斷。",
            "深度結果是相對單目估計，不是真實 3D 距離。",
        ],
        "score_label": "LAION 審美分數",
        "score_hint": "上傳圖片後開始分析",
        "score_sub": "CLIP ViT-L/14 + sa_0_4_vit_l_14_linear.pth · 0-10",
        "dimensions_empty": "上傳後，這裡會顯示拆解維度。",
        "dimensions_title": "拆解代理指標",
        "dimensions_source_prefix": "描述性 · 非規範性 · ",
        "dimensions_disclaimer": "這些條形圖衡量的是「圖中有什麼」，不是「圖是否好看」。請結合 LAION 分數和你自己的審美判斷。",
        "hue_title": "色相與冷暖",
        "hue_hint": "上傳後可查看色環與冷暖條",
        "hue_caption": "色環 · 色相直方圖 · 冷暖條",
        "proxy_empty": "上傳後，這裡會顯示代理視覺化圖。",
        "proxy_title": "代理診斷圖",
        "proxy_desc": "邊緣、顯著性、構圖質心與色板圖，對應上方的拆解條形指標。",
        "cards": {
            "edges": ("邊緣 / 紋理圖", "輪廓與細節結構"),
            "saliency": ("注意力熱力圖", "視線更容易被吸引的位置"),
            "composition": ("構圖網格", "三分法 + 視覺重心"),
            "palette": ("主色板", "粗粒度色彩簇"),
        },
        "depth_unavailable": "深度分層暫不可用：",
        "depth_empty": "上傳後，這裡會顯示前中後景深度分層。",
        "depth_title": "深度圖層分離",
        "depth_desc": "根據單目深度估計，將圖像拆成前景 / 中景 / 背景。",
        "depth_note_prefix": "單目深度模型 ",
        "depth_note_suffix": " · 輸出相對遠近關係，不是真實 3D 距離。",
        "near": "前景",
        "mid": "中景",
        "far": "背景",
        "spread": "深度離散度",
        "depth_cards": {
            "depth_map": ("深度圖", "越亮越靠近鏡頭"),
            "foreground": ("前景（近）", "最近的深度帶"),
            "midground": ("中景", "中間深度帶"),
            "background": ("背景（遠）", "最遠的深度帶"),
        },
        "visual_empty": "上傳圖片後可查看視覺化分析結果。",
        "visual_label": "視覺化分析——代理圖、構圖、邊緣與深度圖層",
        "json_label": "原始指標（JSON）",
        "json_copy_btn": "複製 JSON",
        "json_copied_btn": "已複製",
        "waiting_json": '{\n  "status": "等待上傳圖片"\n}',
        "note": "這些代理條形指標是描述性訊號，不是絕對審美判斷。",
        "analysis_failed": "分析失敗",
    },
}


def _lang_copy(lang: str) -> dict:
    return LANGS.get(lang, LANGS["en"])


def _header_html(lang: str) -> str:
    copy = _lang_copy(lang)
    return f"""
    <header class="app-header">
      <p class="eyebrow">{html.escape(copy["eyebrow"])}</p>
      <h1>{html.escape(copy["title"])}</h1>
      <p class="subtitle">{html.escape(copy["subtitle"])}</p>
    </header>
    """


def _guide_html(lang: str) -> str:
    copy = _lang_copy(lang)
    items = "".join(f"<li>{html.escape(item)}</li>" for item in copy["guide_steps"])
    return f"""
    <section class="intro-card">
      <div class="intro-head">
        <h2>{html.escape(copy["guide_title"])}</h2>
      </div>
      <p class="intro-body">{html.escape(copy["guide_body"])}</p>
      <ul class="intro-list">{items}</ul>
    </section>
    """


def _upload_hint_html(lang: str) -> str:
    return f"<p class='upload-hint'>{html.escape(_lang_copy(lang)['upload_hint'])}</p>"


def _layout_mode(image) -> str:
    if image is None:
        return "empty"
    try:
        return "ready" if _coerce_pil_image(image) is not None else "empty"
    except Exception:
        return "empty"


def _chrome_updates(lang: str, image):
    copy = _lang_copy(lang)
    mode = _layout_mode(image)
    return {
        "header": _header_html(lang),
        "guide": _guide_html(lang),
        "lang": gr.update(label=copy["language_label"]),
        "image": gr.update(label=copy["upload_label"]),
        "btn": gr.update(value=copy["analyze_btn"]),
        "visual_acc": gr.update(label=copy["visual_label"]),
        "json_acc": gr.update(label=copy["json_label"]),
        "hint": _upload_hint_html(lang),
        "main_row": gr.update(elem_classes=["main-row", f"main-row--{mode}"]),
    }


def _score_color(score: float) -> str:
    if score >= 7.0:
        return "#34c759"
    if score >= 5.5:
        return "#ff9500"
    return "#ff3b30"


def _bar_color(v: float) -> str:
    """Neutral intensity ramp — not a good/bad judgment."""
    if v >= 0.72:
        return "#5856d6"
    if v >= 0.45:
        return "#007aff"
    return "#8e8e93"


def _level_badge(level: str) -> str:
    colors = {"high": "#5856d6", "moderate": "#007aff", "low": "#8e8e93"}
    return f'<span class="dim-level" style="color:{colors.get(level, "#8e8e93")}">{html.escape(level)}</span>'


def _render_laion(score: float | None, lang: str) -> str:
    copy = _lang_copy(lang)
    if score is None:
        return f"""
        <div class="score-card sidebar empty">
          <div class="score-label">{html.escape(copy["score_label"])}</div>
          <div class="score-hint">{html.escape(copy["score_hint"])}</div>
        </div>"""
    color = _score_color(score)
    return f"""
    <div class="score-card sidebar">
      <div class="score-label">{html.escape(copy["score_label"])}</div>
      <div class="score-value" style="color:{color}">{score:.2f}</div>
      <div class="score-sub">{html.escape(copy["score_sub"])}</div>
      <div class="score-bar-track">
        <div class="score-bar-fill" style="width:{score/10*100:.0f}%;background:{color}"></div>
      </div>
    </div>"""


def _render_dimensions(dims: list[dict], rationales: list[str], source: str, lang: str) -> str:
    copy = _lang_copy(lang)
    if not dims:
        return f'<div class="dim-empty">{html.escape(copy["dimensions_empty"])}</div>'

    rows = []
    for d, rationale in zip(dims, rationales):
        v = d["value"]
        pct = int(round(v * 100)) if v >= 0.01 else (f"{v * 100:.1f}" if v > 0 else "0")
        if isinstance(pct, float):
            pct_str = f"{pct:.1f}%"
        else:
            pct_str = f"{pct}%"
        color = _bar_color(v)
        level = d.get("level", "")
        interpretation = d.get("interpretation", "")
        sub_rows = ""
        for sub in d.get("sub_metrics", []):
            sv = sub["value"]
            sub_rows += f"""
              <div class="sub-metric">
                <span class="sub-label">{html.escape(sub['label'])}</span>
                <span class="sub-pct">{html.escape(sub.get('pct', f'{sv:.0%}'))}</span>
                <div class="sub-bar-track">
                  <div class="sub-bar-fill" style="width:{int(min(100, sv*100))}%;background:{color}"></div>
                </div>
              </div>"""
        rows.append(
            f"""
            <div class="dim-row">
              <div class="dim-head">
                <span class="dim-name">{html.escape(d['label'])}</span>
                <span class="dim-pct">{pct_str} {_level_badge(level)}</span>
              </div>
              <div class="dim-bar-track">
                <div class="dim-bar-fill" style="width:{max(1, int(v*100)) if v > 0 else 0}%;background:{color}"></div>
              </div>
              <p class="dim-interpret">{html.escape(interpretation)}</p>
              {f'<div class="sub-metrics">{sub_rows}</div>' if sub_rows else ''}
              <p class="dim-rationale">{html.escape(rationale)}</p>
            </div>"""
        )

    return f"""
    <div class="dim-panel">
      <div class="dim-panel-head">
        <span>{html.escape(copy["dimensions_title"])}</span>
        <span class="dim-source">{html.escape(copy["dimensions_source_prefix"] + source)}</span>
      </div>
      <p class="dim-disclaimer">{html.escape(copy["dimensions_disclaimer"])}</p>
      {''.join(rows)}
    </div>"""


def _render_hue_temperature(bundle: dict[str, str] | None, lang: str) -> str:
    copy = _lang_copy(lang)
    if not bundle or not bundle.get("hue_wheel"):
        return f"""
        <div class="score-card sidebar hue-card empty">
          <div class="score-label">{html.escape(copy["hue_title"])}</div>
          <div class="score-hint">{html.escape(copy["hue_hint"])}</div>
        </div>"""
    src = bundle["hue_wheel"]
    return f"""
    <div class="score-card sidebar hue-card">
      <div class="score-label">{html.escape(copy["hue_title"])}</div>
      <figure class="hue-figure">
        <img src="{src}" alt="{html.escape(copy["hue_title"])}" loading="lazy" />
      </figure>
      <p class="hue-caption">{html.escape(copy["hue_caption"])}</p>
    </div>"""


def _render_proxy_diagnostics(bundle: dict[str, str] | None, lang: str) -> str:
    copy = _lang_copy(lang)
    if not bundle:
        return f'<div class="viz-empty">{html.escape(copy["proxy_empty"])}</div>'
    cards = [
        ("edges", *copy["cards"]["edges"]),
        ("saliency", *copy["cards"]["saliency"]),
        ("composition", *copy["cards"]["composition"]),
        ("palette", *copy["cards"]["palette"]),
    ]
    return _render_viz_cards(cards, bundle)


def _render_viz_cards(cards: list[tuple[str, str, str]], bundle: dict[str, str]) -> str:
    items = []
    for key, title, caption in cards:
        src = bundle.get(key, "")
        if not src:
            continue
        items.append(
            f"""
            <figure class="viz-card">
              <img src="{src}" alt="{html.escape(title)}" loading="lazy" />
              <figcaption>
                <strong>{html.escape(title)}</strong>
                <span>{html.escape(caption)}</span>
              </figcaption>
            </figure>"""
        )
    if not items:
        return '<div class="viz-empty">No maps generated.</div>'
    return f'<div class="viz-grid">{"".join(items)}</div>'


def _render_depth_layers(bundle: dict | None, lang: str, error: str | None = None) -> str:
    copy = _lang_copy(lang)
    if error:
        return f'<div class="viz-empty depth-error">{html.escape(copy["depth_unavailable"] + error)}</div>'
    if not bundle:
        return f'<div class="viz-empty">{html.escape(copy["depth_empty"])}</div>'

    stats = bundle.get("stats", {})
    cards = [
        ("depth_map", *copy["depth_cards"]["depth_map"]),
        ("foreground", *copy["depth_cards"]["foreground"]),
        ("midground", *copy["depth_cards"]["midground"]),
        ("background", *copy["depth_cards"]["background"]),
    ]
    grid = _render_viz_cards(cards, bundle)

    fg = stats.get("foreground_coverage", 0)
    mg = stats.get("midground_coverage", 0)
    bg = stats.get("background_coverage", 0)
    spread = stats.get("depth_spread", 0)
    model = stats.get("model", "DPT")

    return f"""
    <div class="depth-panel">
      <p class="depth-note">
        {html.escape(copy["depth_note_prefix"])}<code>{html.escape(model)}</code>{html.escape(copy["depth_note_suffix"])}
      </p>
      <div class="depth-stats">
        <span>{html.escape(copy["near"])} <strong>{fg:.0%}</strong></span>
        <span>{html.escape(copy["mid"])} <strong>{mg:.0%}</strong></span>
        <span>{html.escape(copy["far"])} <strong>{bg:.0%}</strong></span>
        <span>{html.escape(copy["spread"])} <strong>{spread:.2f}</strong></span>
      </div>
      {grid}
    </div>"""


def _render_all_visuals(
    diagnostics: dict[str, str] | None,
    depth_layers: dict | None,
    lang: str,
    depth_error: str | None = None,
) -> str:
    """Single panel: proxy diagnostics + depth layers (nothing removed)."""
    copy = _lang_copy(lang)
    if not diagnostics and not depth_layers and not depth_error:
        return f'<div class="viz-empty">{html.escape(copy["visual_empty"])}</div>'

    return f"""
    <div class="visual-analysis">
      <section class="viz-section">
        <h3 class="viz-section-title">{html.escape(copy["proxy_title"])}</h3>
        <p class="viz-section-desc">{html.escape(copy["proxy_desc"])}</p>
        {_render_proxy_diagnostics(diagnostics, lang)}
      </section>
      <section class="viz-section">
        <h3 class="viz-section-title">{html.escape(copy["depth_title"])}</h3>
        <p class="viz-section-desc">{html.escape(copy["depth_desc"])}</p>
        {_render_depth_layers(depth_layers, lang, depth_error)}
      </section>
    </div>"""


def _render_json(payload: str, lang: str) -> str:
    copy = _lang_copy(lang)
    copy_label = html.escape(copy["json_copy_btn"])
    copied_label = html.escape(copy["json_copied_btn"])
    return f"""
    <div class="json-panel">
      <div class="json-toolbar">
        <button
          type="button"
          class="json-copy-btn"
          data-copy-label="{copy_label}"
          data-copied-label="{copied_label}"
          onclick="window.copyJsonMetrics && window.copyJsonMetrics(this)"
        >{copy_label}</button>
      </div>
      <pre class="json-raw">{html.escape(payload)}</pre>
    </div>"""


def _render_error(message: str, lang: str) -> str:
    copy = _lang_copy(lang)
    return f"""
    <div class="score-card empty">
      <div class="score-label">{html.escape(copy["analysis_failed"])}</div>
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


def _render_page(image, lang: str):
    copy = _lang_copy(lang)
    waiting = copy["waiting_json"]
    empty_visuals = _render_all_visuals(None, None, lang)
    empty_hue = _render_hue_temperature(None, lang)
    chrome = _chrome_updates(lang, image)
    if image is None:
        return (
            chrome["header"],
            chrome["guide"],
            chrome["lang"],
            chrome["image"],
            chrome["btn"],
            chrome["visual_acc"],
            chrome["json_acc"],
            chrome["hint"],
            chrome["main_row"],
            _render_laion(None, lang),
            empty_hue,
            _render_dimensions([], [], "", lang),
            _render_json(waiting, lang),
            empty_visuals,
        )

    try:
        pil = _coerce_pil_image(image)
        if pil is None:
            return (
                chrome["header"],
                chrome["guide"],
                chrome["lang"],
                chrome["image"],
                chrome["btn"],
                chrome["visual_acc"],
                chrome["json_acc"],
                chrome["hint"],
                chrome["main_row"],
                _render_laion(None, lang),
                empty_hue,
                _render_dimensions([], [], "", lang),
                _render_json(waiting, lang),
                empty_visuals,
            )

        laion = laion_aesthetic_score(pil)
        dims = compute_dissection(pil)
        rationales, source = generate_rationales(dims, laion)
        diagnostics = build_diagnostic_bundle(pil)

        depth_layers = None
        depth_error = None
        try:
            depth_layers = separate_depth_layers(pil)
        except Exception as depth_exc:
            depth_error = str(depth_exc)

        detail = {
            "laion_aesthetic_score": laion,
            "device": str(DEVICE),
            "rationale_source": source,
            "note": copy["note"],
            "dimensions": [{**d, "rationale": r} for d, r in zip(dims, rationales)],
            "depth_layers": depth_layers.get("stats", {}) if depth_layers else None,
            "depth_error": depth_error,
        }
        chrome = _chrome_updates(lang, image)
        return (
            chrome["header"],
            chrome["guide"],
            chrome["lang"],
            chrome["image"],
            chrome["btn"],
            chrome["visual_acc"],
            chrome["json_acc"],
            chrome["hint"],
            chrome["main_row"],
            _render_laion(laion, lang),
            _render_hue_temperature(diagnostics, lang),
            _render_dimensions(dims, rationales, source, lang),
            _render_json(json.dumps(detail, indent=2), lang),
            _render_all_visuals(diagnostics, depth_layers, lang, depth_error),
        )
    except Exception as exc:
        err = str(exc)
        return (
            chrome["header"],
            chrome["guide"],
            chrome["lang"],
            chrome["image"],
            chrome["btn"],
            chrome["visual_acc"],
            chrome["json_acc"],
            chrome["hint"],
            chrome["main_row"],
            _render_error(err, lang),
            empty_hue,
            _render_dimensions([], [], "", lang),
            _render_json(json.dumps({"error": err}, indent=2), lang),
            empty_visuals,
        )


with gr.Blocks(
    title="Aesthetic Dissection Panel",
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="gray", neutral_hue="gray"),
    css=CUSTOM_CSS,
) as demo:
    default_lang = "en"
    with gr.Column(elem_classes=["page-top"]):
        lang_in = gr.Dropdown(
            choices=[
                ("English", "en"),
                ("中文简体", "zh-Hans"),
                ("中文繁體", "zh-Hant"),
            ],
            value=default_lang,
            label=_lang_copy(default_lang)["language_label"],
            elem_classes=["lang-select"],
            scale=0,
            min_width=140,
            container=False,
        )
        header_out = gr.HTML(_header_html(default_lang))

    guide_out = gr.HTML(_guide_html(default_lang))

    with gr.Row(elem_classes=["main-row", "main-row--empty"]) as main_row:
        with gr.Column(scale=5, elem_classes=["left-col"]):
            image_in = gr.Image(type="pil", label=_lang_copy(default_lang)["upload_label"], height=480)
            upload_hint_out = gr.HTML(_upload_hint_html(default_lang))
            analyze_btn = gr.Button(_lang_copy(default_lang)["analyze_btn"], variant="primary")
            laion_out = gr.HTML(_render_laion(None, default_lang))
            hue_out = gr.HTML(_render_hue_temperature(None, default_lang))
        with gr.Column(scale=6, elem_classes=["right-col"]):
            dims_out = gr.HTML(_render_dimensions([], [], "", default_lang))

    with gr.Accordion(_lang_copy(default_lang)["visual_label"], open=True) as visual_acc:
        visual_out = gr.HTML(_render_all_visuals(None, None, default_lang))

    with gr.Accordion(_lang_copy(default_lang)["json_label"], open=False) as json_acc:
        json_out = gr.HTML(_render_json(_lang_copy(default_lang)["waiting_json"], default_lang))

    gr.HTML(
        """
        <script>
        if (!window.copyJsonMetrics) {
          window.copyJsonMetrics = function(btn) {
            const panel = btn.closest(".json-panel");
            if (!panel) return;
            const pre = panel.querySelector(".json-raw");
            if (!pre) return;
            const copyLabel = btn.dataset.copyLabel || "Copy JSON";
            const copiedLabel = btn.dataset.copiedLabel || "Copied!";
            const text = pre.innerText || pre.textContent || "";
            const done = function() {
              btn.textContent = copiedLabel;
              btn.classList.add("copied");
              setTimeout(function() {
                btn.textContent = copyLabel;
                btn.classList.remove("copied");
              }, 1600);
            };
            if (navigator.clipboard && navigator.clipboard.writeText) {
              navigator.clipboard.writeText(text).then(done).catch(function() {
                const ta = document.createElement("textarea");
                ta.value = text;
                document.body.appendChild(ta);
                ta.select();
                document.execCommand("copy");
                document.body.removeChild(ta);
                done();
              });
            } else {
              const ta = document.createElement("textarea");
              ta.value = text;
              document.body.appendChild(ta);
              ta.select();
              document.execCommand("copy");
              document.body.removeChild(ta);
              done();
            }
          };
        }
        </script>
        """,
        visible=False,
    )

    page_outputs = [
        header_out,
        guide_out,
        lang_in,
        image_in,
        analyze_btn,
        visual_acc,
        json_acc,
        upload_hint_out,
        main_row,
        laion_out,
        hue_out,
        dims_out,
        json_out,
        visual_out,
    ]
    analyze_event = dict(fn=_render_page, inputs=[image_in, lang_in], outputs=page_outputs)
    image_in.change(**analyze_event)
    analyze_btn.click(**analyze_event)
    lang_in.change(**analyze_event)

    demo.queue(default_concurrency_limit=1)


if __name__ == "__main__":
    print(f"\n🎨  Aesthetic Dissection Panel · device = {DEVICE}\n")
    demo.launch(share=SHARE_LOCAL)
