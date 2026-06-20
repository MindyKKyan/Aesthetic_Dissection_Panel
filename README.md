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

**Turn black-box aesthetic scores into auditable dimensions.**

Upload any image and get:

1. **LAION Aesthetic Score** (0–10) — holistic baseline from CLIP ViT-L/14 + `sa_0_4_vit_l_14_linear.pth`
2. **5 Dissection Sliders** — diagnostic proxies for:
   - 🎨 **Color Harmony** — HSV hue entropy × luminance contrast ratio
   - ⚖️ **Composition Balance** — saliency / luminance center-of-mass offset
   - 🌈 **Saturation Intensity** — mean HSV saturation
   - 🧩 **Edge Complexity** — Canny edge density
   - 🌡️ **Warm / Cool** — circular hue centroid
3. **Plain-English Rationales** — each slider includes an explanation of what the value means for your image (GPT-4o mini → HF Inference → rule-based template fallback)

## Why this exists

In image curation workflows (ComfyUI, generative pipelines, etc.), **aesthetic judgment is a black box**. This panel is my answer: *operationalize aesthetic quality into inspectable, auditable dimensions.*

## Research Context

This project speaks to **HCI + Aesthetics + XAI-light**:

- Not just an accept/reject score
- A **dimension-level diagnostic** that makes aesthetic reasoning transparent
- Designed for researchers and practitioners who want to understand *why* an image works (or doesn't)

## Stack

- Python · Gradio 5 · PyTorch (MPS / CPU / CUDA auto)
- Hugging Face Transformers (CLIP ViT-L/14) · LAION aesthetic predictor weights
- OpenCV (proxy metrics) · GPT-4o mini / HF Inference (optional rationales)

## Try it (online)

Open the Hugging Face Space — upload an image, see the LAION score, explore the sliders, read the rationales. No install required.

## Run locally

```bash
git clone git@github.com:MindyKKyan/Aesthetic_Dissection_Panel.git
cd Aesthetic_Dissection_Panel
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # installs packages/aesthetic_core automatically
export PYTORCH_ENABLE_MPS_FALLBACK=1   # optional, M3 Mac
python app.py
# → http://127.0.0.1:7860
```

First run downloads CLIP ViT-L/14 weights (~900 MB) and the LAION linear head automatically.

Optional LLM rationales: set `OPENAI_API_KEY` or `HF_TOKEN` in your environment (or HF Space Secrets).

---
© 2026 — research demonstration prototype.
