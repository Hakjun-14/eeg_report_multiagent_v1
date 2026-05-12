#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY: present"
  exit 0
fi

if [[ -n "${GPT_API_KEY:-}" ]]; then
  echo "GPT_API_KEY: present (will be mapped to OPENAI_API_KEY by start_container.sh)"
  exit 0
fi

echo "API key missing: set OPENAI_API_KEY (or GPT_API_KEY) in your shell"
exit 1
