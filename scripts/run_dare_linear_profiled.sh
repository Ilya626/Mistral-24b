#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Ensure our Python patches are picked up before invoking mergekit.
export PYTHONPATH="${REPO_ROOT}/pythonpath${PYTHONPATH:+:${PYTHONPATH}}"

CFG="${CFG:-${REPO_ROOT}/configs/dare_linear_profiled.yml}"
OUT="${OUT:-/workspace/merged_dare_linear_profiled}"

VISTRAL_DIR="${VISTRAL_DIR:-/workspace/Vistral-24B-Instruct}"
CYDONIA_DIR="${CYDONIA_DIR:-/workspace/Cydonia-24B-v4.2.0}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-/workspace/Mistral-Small-3.2-24B-Instruct-2506}"

ensure_path_in_config() {
  local label="$1"
  local path="$2"
  if [[ ! -f "${CFG}" ]]; then
    echo "Конфиг mergekit не найден: ${CFG}" >&2
    exit 1
  fi
  if ! grep -Fq -- "${path}" "${CFG}"; then
    echo "Конфиг ${CFG} не содержит путь для ${label}: ${path}" >&2
    echo "Обновите YAML или задайте переменную окружения, чтобы пути совпадали." >&2
    exit 1
  fi
}

ensure_path_in_config "Vistral" "${VISTRAL_DIR}"
ensure_path_in_config "Cydonia" "${CYDONIA_DIR}"
ensure_path_in_config "базовой модели" "${BASE_MODEL_DIR}"

echo "--- ФАЗА: DARE-Linear (профиль) ---"

[[ -z "${HF_TOKEN:-}" ]] && { echo "HF_TOKEN не задан"; exit 1; }

echo ">>> Загружаю модели..."
huggingface-cli download Vikhrmodels/Vistral-24B-Instruct \
  --local-dir "${VISTRAL_DIR}" --exclude "*.bin"
huggingface-cli download TheDrummer/Cydonia-24B-v4.2.0 \
  --local-dir "${CYDONIA_DIR}"
huggingface-cli download mistralai/Mistral-Small-3.2-24B-Instruct-2506 \
  --local-dir "${BASE_MODEL_DIR}" --exclude "*.bin"

echo ">>> Запускаю mergekit (dare_linear + профили)..."
mergekit-yaml "${CFG}" "${OUT}" \
  --cuda --copy-tokenizer --clone-tensors

echo "--- Готово ---"
echo "Результат: ${OUT}"
