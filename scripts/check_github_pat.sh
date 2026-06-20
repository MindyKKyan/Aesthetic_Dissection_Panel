#!/usr/bin/env bash
# Test whether GITHUB_PAT can write to the repo (run in Terminal.app).
# Usage: export GITHUB_PAT="ghp_..." && bash scripts/check_github_pat.sh

set -euo pipefail

if [[ -z "${GITHUB_PAT:-}" ]]; then
  echo "Set GITHUB_PAT first, e.g.:"
  echo '  export GITHUB_PAT="ghp_xxxx"'
  exit 1
fi

REPO="MindyKKyan/Aesthetic_Dissection_Panel"
API="https://api.github.com/repos/${REPO}"

echo "==> Who am I?"
curl -sS -H "Authorization: Bearer ${GITHUB_PAT}" \
  -H "Accept: application/vnd.github+json" \
  https://api.github.com/user | python3 -c "import sys,json; u=json.load(sys.stdin); print('  login:', u.get('login','?'), '| type:', u.get('type','?'))"

echo "==> Repo access (GET ${REPO})"
HTTP=$(curl -sS -o /tmp/gh_repo.json -w "%{http_code}" \
  -H "Authorization: Bearer ${GITHUB_PAT}" \
  -H "Accept: application/vnd.github+json" \
  "$API")
echo "  HTTP $HTTP"
python3 -c "
import json
d=json.load(open('/tmp/gh_repo.json'))
if 'message' in d:
    print('  error:', d['message'])
else:
    print('  full_name:', d.get('full_name'))
    print('  private:', d.get('private'))
    print('  permissions:', d.get('permissions'))
" 2>/dev/null || cat /tmp/gh_repo.json

echo "==> Push permission hint"
PERMS=$(python3 -c "import json; d=json.load(open('/tmp/gh_repo.json')); print(d.get('permissions',{}))" 2>/dev/null || echo "{}")
if echo "$PERMS" | grep -q "'push': True"; then
  echo "  ✅ push: True — PAT should work. Re-run: bash scripts/deploy.sh"
elif echo "$PERMS" | grep -q "'push': False"; then
  echo "  ❌ push: False — token is READ-ONLY. Create new PAT with Contents: Read and write (or classic scope 'repo')."
else
  echo "  ❌ No repo access. Fine-grained PAT must select repository: Aesthetic_Dissection_Panel"
  echo "     Classic PAT must include scope: repo"
fi
