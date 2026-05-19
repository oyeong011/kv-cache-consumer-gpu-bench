#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 2 ]; then
  echo "usage: $0 <github-owner> <repo-name>" >&2
  echo "example: $0 oyeong011 kv-cache-consumer-gpu-bench" >&2
  exit 2
fi
OWNER="$1"
REPO="$2"
cd "$(dirname "$0")/.."
if ! command -v gh >/dev/null 2>&1; then
  echo "gh CLI not found. Install GitHub CLI and run: gh auth login" >&2
  exit 1
fi
if [ ! -d .git ]; then
  git init
fi
git add .
if ! git diff --cached --quiet; then
  git commit -m "Measure KV-cache pressure on consumer GPUs

Constraint: Consumer GPUs cannot reproduce Oaken's dedicated hardware path.
Rejected: Claiming Oaken performance reproduction | This benchmark measures the root KV-cache pressure instead.
Confidence: high
Scope-risk: narrow
Directive: Interpret results as empirical cache pressure and strategy trade-offs, not Oaken reproduction.
Tested: Python syntax compilation for benchmark and analysis scripts.
Not-tested: Full CUDA benchmark requires local GPU, model downloads, and optional quantized-cache backends."
fi
git branch -M main
if ! gh repo view "$OWNER/$REPO" >/dev/null 2>&1; then
  gh repo create "$OWNER/$REPO" --public --source=. --remote=origin --push
else
  git remote remove origin 2>/dev/null || true
  git remote add origin "https://github.com/$OWNER/$REPO.git"
  git push -u origin main
fi
