#!/usr/bin/env bash
# Upload Panel to HF Space only (skip GitHub). Requires: hf auth login
set -euo pipefail
PANEL="$(cd "$(dirname "$0")/.." && pwd)"
SPACE_ID="Mindykkyan/Aesthetic_Dissection_Panel"

hf auth whoami || { echo "Run: hf auth login"; exit 1; }

cd "$PANEL"
hf upload "$SPACE_ID" . . \
  --repo-type space \
  --exclude ".git/**" \
  --exclude "scripts/**" \
  --exclude "weights/**" \
  --exclude "**/__pycache__/**" \
  --exclude ".gitignore" \
  --commit-message "feat: Aesthetic Dissection Panel"

echo "✅ https://huggingface.co/spaces/${SPACE_ID}"
