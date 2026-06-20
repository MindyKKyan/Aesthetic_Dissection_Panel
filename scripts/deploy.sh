#!/usr/bin/env bash
# Run in Terminal.app (NOT Cursor terminal) to avoid vscode-git credential helper issues.
#
# GitHub:
#   export GITHUB_PAT="ghp_xxxx"   # classic PAT with `repo` scope
#
# Hugging Face (pick one):
#   hf auth login                  # browser OAuth — works with `hf upload` below
#   export HF_TOKEN="hf_xxxx"      # Write token from huggingface.co/settings/tokens
#
# Usage:
#   bash scripts/deploy.sh

set -euo pipefail

PANEL="$(cd "$(dirname "$0")/.." && pwd)"
SPACE_ID="Mindykkyan/Aesthetic_Dissection_Panel"

echo "==> Panel source: $PANEL"

# --- GitHub ---
if [[ -z "${GITHUB_PAT:-}" ]]; then
  echo "ERROR: Set GITHUB_PAT first (GitHub → Settings → Developer settings → Personal access tokens)"
  echo "  Classic: enable 'repo' scope"
  exit 1
fi

cd "$PANEL"
git -c credential.helper= push "https://MindyKKyan:${GITHUB_PAT}@github.com/MindyKKyan/Aesthetic_Dissection_Panel.git" main:main
git remote set-url origin https://github.com/MindyKKyan/Aesthetic_Dissection_Panel.git
echo "✅ GitHub push OK"

# --- Hugging Face Space (via hf upload — works with `hf auth login` OAuth) ---
if ! command -v hf >/dev/null 2>&1; then
  echo "ERROR: install huggingface_hub CLI: pip install huggingface_hub"
  exit 1
fi

if ! hf auth whoami >/dev/null 2>&1; then
  echo "HF: run 'hf auth login' first (or export HF_TOKEN with Write access)"
  exit 1
fi

echo "==> Uploading to HF Space: $SPACE_ID"
cd "$PANEL"
hf upload "$SPACE_ID" . . \
  --repo-type space \
  --exclude ".git/**" \
  --exclude "scripts/**" \
  --exclude "weights/**" \
  --exclude "**/__pycache__/**" \
  --exclude ".gitignore" \
  --commit-message "feat: Aesthetic Dissection Panel"

echo "✅ Hugging Face Space upload OK"
echo "   https://huggingface.co/spaces/${SPACE_ID}"
echo "   (Build may take 3–5 min on first run while CLIP weights download)"
