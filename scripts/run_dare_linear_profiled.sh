#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Ensure our Python patches are picked up before invoking mergekit.
export PYTHONPATH="${REPO_ROOT}/pythonpath${PYTHONPATH:+:${PYTHONPATH}}"

CFG="${REPO_ROOT}/configs/dare_linear_profiled.yml"
OUT="/workspace/merged_dare_linear_profiled"

echo "--- ФАЗА: DARE-Linear (профиль) ---"

[[ -z "${HF_TOKEN:-}" ]] && { echo "HF_TOKEN не задан"; exit 1; }

echo ">>> Загружаю модели..."
huggingface-cli download Vikhrmodels/Vistral-24B-Instruct \
  --local-dir /workspace/Vistral-24B-Instruct --exclude "*.bin"
huggingface-cli download TheDrummer/Cydonia-24B-v4.2.0 \
  --local-dir /workspace/Cydonia-24B-v4.2.0
huggingface-cli download mistralai/Mistral-Small-3.2-24B-Instruct-2506 \
  --local-dir /workspace/Mistral-Small-3.2-24B-Instruct-2506 --exclude "*.bin"

echo ">>> Запускаю mergekit (dare_linear + профили)..."
mergekit-yaml "${CFG}" "${OUT}" \
  --cuda --copy-tokenizer --clone-tensors

echo "--- Готово ---"
echo "Результат: ${OUT}"
