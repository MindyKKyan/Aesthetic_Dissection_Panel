#!/usr/bin/env bash
# Run in Terminal.app (NOT Cursor terminal) to avoid vscode-git credential helper issues.
#
# Usage:
#   export GITHUB_PAT="ghp_xxxx"   # classic PAT with `repo` scope, or fine-grained with Contents: write
#   bash scripts/deploy.sh

set -euo pipefail

PANEL="$(cd "$(dirname "$0")/.." && pwd)"
HF_CLONE="${HF_CLONE:-/tmp/hf-adp}"

echo "==> Panel source: $PANEL"

# --- GitHub ---
if [[ -z "${GITHUB_PAT:-}" ]]; then
  echo "ERROR: Set GITHUB_PAT first (GitHub → Settings → Developer settings → Personal access tokens)"
  echo "  Classic: enable 'repo' scope"
  echo "  Fine-grained: select repo Aesthetic_Dissection_Panel + Contents Read/Write"
  exit 1
fi

cd "$PANEL"
git -c credential.helper= push "https://MindyKKyan:${GITHUB_PAT}@github.com/MindyKKyan/Aesthetic_Dissection_Panel.git" main:main
git remote set-url origin https://github.com/MindyKKyan/Aesthetic_Dissection_Panel.git
echo "✅ GitHub push OK"

# --- Hugging Face Space ---
HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)}"
if [[ -z "$HF_TOKEN" ]]; then
  echo "HF: run 'hf auth login' first, then re-run this script"
  exit 0
fi

rm -rf "$HF_CLONE"
git clone "https://Mindykkyan:${HF_TOKEN}@huggingface.co/spaces/Mindykkyan/Aesthetic_Dissection_Panel" "$HF_CLONE"
cd "$HF_CLONE"

cp "$PANEL/app.py" "$PANEL/requirements.txt" "$PANEL/README.md" .
rm -rf css src
cp -r "$PANEL/css" "$PANEL/src" .
rm -rf src/__pycache__

git add .
git diff --staged --quiet || git commit -m "feat: Aesthetic Dissection Panel"
git -c credential.helper= push "https://Mindykkyan:${HF_TOKEN}@huggingface.co/spaces/Mindykkyan/Aesthetic_Dissection_Panel" main:main
echo "✅ Hugging Face Space push OK"
echo "   https://huggingface.co/spaces/Mindykkyan/Aesthetic_Dissection_Panel"
