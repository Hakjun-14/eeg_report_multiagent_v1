#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
HOST_WORKSPACE="$(dirname "${PROJECT_DIR}")"
INSTALL_EVAL_DEPS="${INSTALL_EVAL_DEPS:-0}"
INSTALL_BERTSCORE_DEPS="${INSTALL_BERTSCORE_DEPS:-0}"
IMAGE_TAG="latest"
if [[ "${INSTALL_EVAL_DEPS}" == "1" && "${INSTALL_BERTSCORE_DEPS}" == "1" ]]; then
  IMAGE_TAG="eval-bertscore"
elif [[ "${INSTALL_EVAL_DEPS}" == "1" ]]; then
  IMAGE_TAG="eval"
fi
IMAGE_NAME="eeg-report-multiagent-v1:${IMAGE_TAG}"
CONTAINER_NAME="eeg-report-multiagent-v1"

if ! command -v docker >/dev/null 2>&1; then
  echo "[error] docker command not found"
  exit 1
fi

if [[ -z "${OPENAI_API_KEY:-}" && -n "${GPT_API_KEY:-}" ]]; then
  export OPENAI_API_KEY="${GPT_API_KEY}"
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "[error] OPENAI_API_KEY (or GPT_API_KEY) is not set in host shell"
  echo "        example: export OPENAI_API_KEY='sk-...'"
  exit 1
fi

echo "[info] Building image: ${IMAGE_NAME} (INSTALL_EVAL_DEPS=${INSTALL_EVAL_DEPS}, INSTALL_BERTSCORE_DEPS=${INSTALL_BERTSCORE_DEPS})"
docker build \
  --build-arg INSTALL_EVAL_DEPS="${INSTALL_EVAL_DEPS}" \
  --build-arg INSTALL_BERTSCORE_DEPS="${INSTALL_BERTSCORE_DEPS}" \
  -t "${IMAGE_NAME}" \
  "${PROJECT_DIR}"

echo "[info] Starting container: ${CONTAINER_NAME}"
docker run --rm -it \
  --name "${CONTAINER_NAME}" \
  -e OPENAI_API_KEY \
  -v "${HOST_WORKSPACE}:/workspace" \
  -w "/workspace/$(basename "${PROJECT_DIR}")" \
  "${IMAGE_NAME}" \
  bash
