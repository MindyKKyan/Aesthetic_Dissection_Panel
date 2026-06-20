---
title: Aesthetic Dissection Panel
emoji: 🔬
colorFrom: indigo
colorTo: red
sdk: gradio
sdk_version: "5.10.0"
app_file: app.py
pinned: false
license: apache-2.0
---

# 🔬 Aesthetic Dissection Panel

**Turn black-box aesthetic scores into auditable dimensions.**

Upload any image and get:

1. **LAION Aesthetic Score** (0–10) — a holistic baseline from CLIP ViT-L/14 + `sa_0_4_vit_l_14_linear.pth`
2. **5 Dissection Sliders** — diagnostic proxies for:
   - 🎨 **Color Harmony** — HSV histogram spread / contrast ratio
   - ⚖️ **Composition Balance** — saliency center-of-mass offset
   - 🌈 **Saturation Intensity** — mean saturation
   - 🧩 **Edge Complexity** — Canny density
   - 🌡️ **Warm/Cool** — hue centroid
3. **Plain-English Rationales** — each slider comes with an explanation of what the value means for your image (GPT-4o-mini → HF Inference → template fallback)

## Why this exists

In image curation workflows (ComfyUI, generative pipelines, etc.), **aesthetic judgment is a black box**. This panel is my answer: *operationalize aesthetic quality into inspectable, auditable dimensions.*

## Research Context

This project speaks to **HCI + Aesthetics + XAI-light**:

- Not just an accept/reject score
- A **dimension-level diagnostic** that makes aesthetic reasoning transparent
- Designed for researchers and practitioners who want to understand *why* an image works (or doesn't)

## Stack

- Python · Gradio 5 · PyTorch · Transformers (CLIP ViT-L/14)
- LAION aesthetic predictor weights (auto-download)
- OpenCV (proxy metrics)
- GPT-4o-mini or HF Inference (optional rationales)

## Try it

Upload an image → see the LAION score → explore the dimension sliders → read the rationales.

## Run locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTORCH_ENABLE_MPS_FALLBACK=1   # M3 Mac
python app.py
```

Optional: set `OPENAI_API_KEY` or `HF_TOKEN` for LLM-generated rationales.

## Live Demo

https://huggingface.co/spaces/Mindykkyan/Aesthetic_Dissection_Panel

---
© 2026 — research demonstration prototype.
