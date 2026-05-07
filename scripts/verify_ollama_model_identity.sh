#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/verify_ollama_model_identity.sh <model>

Environment overrides:
  OLLAMA_HOST_URL   Default: http://127.0.0.1:11434
  OUT_ROOT          Default: outputs/model-identity

The script captures `ollama show` output so we can map local serving aliases to
real upstream checkpoints before using them in paper-facing claims.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

MODEL="${1:-}"
if [[ -z "$MODEL" ]]; then
  usage
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OLLAMA_HOST_URL="${OLLAMA_HOST_URL:-http://127.0.0.1:11434}"
OLLAMA_CLI_HOST="${OLLAMA_HOST_URL#http://}"
OUT_ROOT="${OUT_ROOT:-outputs/model-identity}"
TIMESTAMP="$(date -u +"%Y%m%dT%H%M%SZ")"
SAFE_NAME="${MODEL//[:\/]/-}"
OUTPUT_PATH="${ROOT}/${OUT_ROOT}/${TIMESTAMP}-${SAFE_NAME}.txt"

mkdir -p "$(dirname "$OUTPUT_PATH")"

if ! curl -fsS "${OLLAMA_HOST_URL}/api/tags" >/dev/null; then
  echo "Ollama is not reachable at ${OLLAMA_HOST_URL}." >&2
  exit 2
fi

{
  echo "captured_at_utc=${TIMESTAMP}"
  echo "ollama_host_url=${OLLAMA_HOST_URL}"
  echo "ollama_cli_host=${OLLAMA_CLI_HOST}"
  echo "model=${MODEL}"
  echo
  echo "=== ollama show ${MODEL} ==="
  OLLAMA_HOST="${OLLAMA_CLI_HOST}" ollama show "${MODEL}" || true
  echo
  echo "=== ollama show --modelfile ${MODEL} ==="
  OLLAMA_HOST="${OLLAMA_CLI_HOST}" ollama show --modelfile "${MODEL}" || true
} >"${OUTPUT_PATH}"

echo "[identity] wrote ${OUTPUT_PATH}"
