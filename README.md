---
title: Aesthetic Dissection Panel
emoji: 🔬
colorFrom: indigo
colorTo: red
sdk: gradio
sdk_version: "5.50.0"
app_file: app.py
python_version: "3.11"
pinned: false
license: apache-2.0
---

# 🔬 Aesthetic Dissection Panel

**Turn black-box aesthetic scores into auditable, visual dimensions.**

Upload an image → get a LAION baseline score, five descriptive proxy bars, inspectable diagnostic maps, hue/temperature analysis, and monocular depth layer separation. UI supports **English · 简体中文 · 繁體中文**.

**Live demo:** [Hugging Face Space](https://huggingface.co/spaces/Mindykkyan/Aesthetic_Dissection_Panel)  
**Source:** [GitHub](https://github.com/MindyKKyan/Aesthetic_Dissection_Panel)

## What you get

### 1. LAION Aesthetic Score (0–10)

Holistic baseline from **CLIP ViT-L/14** + `sa_0_4_vit_l_14_linear.pth`.

### 2. Five dissection proxies (descriptive, not prescriptive)

Bars measure *what is present in the image*, not universal “good/bad” judgments:

| Proxy | What it measures |
|-------|------------------|
| **Color Structure** | Hue diversity + luminance contrast |
| **Visual Mass / Composition** | Saliency center-of-mass vs. image center |
| **Color Vividness** | Mean saturation (muted → vivid) |
| **Visual Texture / Detail** | Canny edge density |
| **Color Temperature** | Warm vs. cool hue bias |

Each bar includes sub-metrics, a level badge (low / moderate / high), and a plain-English rationale (GPT-4o mini → HF Inference → rule-based template fallback).

### 3. Visual diagnostics

- **Edge / texture map** — where fine structure appears  
- **Attention heatmap** — saliency overlay  
- **Composition grid** — rule-of-thirds + visual mass crosshair  
- **Dominant palette** — coarse color clusters  
- **Hue & temperature** — color ring, polar hue histogram, warm/cool bar (shown beside LAION score)

### 4. Depth layer separation

Monocular depth via **Intel DPT (`dpt-hybrid-midas`)** splits the image into:

- Depth map (brighter = nearer)  
- Foreground / midground / background layers  

Relative depth only — not metric 3D reconstruction.

### 5. Raw JSON + one-click copy

Full metrics export at the bottom, with a **Copy JSON** button for downstream workflows.

## Why this exists

In image curation workflows (ComfyUI, generative pipelines, etc.), **aesthetic judgment is often a black box**. This panel operationalizes visual quality into **inspectable, auditable signals** — closer to InkIdeator / HEAI-style dimension diagnostics than a single accept/reject score.

## Research context

**HCI · Aesthetics · XAI-light**

- Not just an accept/reject score  
- Dimension-level diagnostics with linked visual evidence  
- Descriptive proxies you can pair with your own taste and task context  

## Stack

- Python · **Gradio 5.50** · PyTorch (MPS / CUDA / CPU auto)  
- Hugging Face Transformers — CLIP ViT-L/14, Intel DPT depth  
- OpenCV — proxy metrics & diagnostic maps  
- Optional: OpenAI / HF Inference for rationales  

## Project layout

```
AestheticDissectionPanel/
├── app.py                 # Gradio UI
├── aesthetic_core/        # CLIP, LAION scorer, dissection, depth, viz
├── css/custom.css         # Apple-minimal theme
├── src/rationale.py       # LLM / template rationales
└── scripts/deploy*.sh     # GitHub + HF deploy helpers
```

## Run locally

```bash
git clone https://github.com/MindyKKyan/Aesthetic_Dissection_Panel.git
cd Aesthetic_Dissection_Panel
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTORCH_ENABLE_MPS_FALLBACK=1   # optional, Apple Silicon
python app.py
# → http://127.0.0.1:7860
```

**First run** downloads model weights automatically:

- CLIP ViT-L/14 (~900 MB) + LAION linear head  
- Intel DPT depth model (~400 MB, on first depth analysis)  

**Optional rationales:** set `OPENAI_API_KEY` or `HF_TOKEN` in your environment (or HF Space Secrets).

## Deploy

```bash
# GitHub + Hugging Face (Terminal.app recommended)
export GITHUB_PAT="ghp_xxxx"   # classic PAT, repo scope
hf auth login
bash scripts/deploy.sh

# Hugging Face only
bash scripts/deploy_hf_only.sh
```

## Language

Use the dropdown in the top-right corner: **English · 中文简体 · 中文繁體**. Switching language updates labels, guides, and empty-state copy without re-uploading.

---
© 2026 — research demonstration prototype.
