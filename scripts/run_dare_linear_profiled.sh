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
    echo "Ошибка: mergekit не нашёл конфиг: ${CFG}" >&2
    exit 1
  fi
  if ! grep -Fq -- "${path}" "${CFG}"; then
    echo "Ошибка: в ${CFG} нет ожидаемого пути для ${label}: ${path}" >&2
    echo "Проверьте YAML: пропишите абсолютные пути и повторите запуск." >&2
    exit 1
  fi
}

ensure_path_in_config "Vistral" "${VISTRAL_DIR}"
ensure_path_in_config "Cydonia" "${CYDONIA_DIR}"
ensure_path_in_config "базовой модели" "${BASE_MODEL_DIR}"

verify_layer_ranges() {
  local overrides
  if ! overrides="$(python3 - "${CFG}" "${BASE_MODEL_DIR}" <<'PY'
import ast
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
base_model_path = Path(sys.argv[2])
expected = 40
current_model = None
overrides: list[str] = []

def _normalise_model(value: str | None) -> str:
    if not value:
        return ""
    return str(Path(value).expanduser().resolve())


with config_path.open("r", encoding="utf-8") as handle:
    for raw_line in handle:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("- model:"):
            suffix = line.split(":", 1)[1].strip()
            if suffix.startswith("&"):
                parts = suffix.split(None, 1)
                suffix = parts[1] if len(parts) == 2 else ""
            current_model = suffix or None
            continue
        if not line.startswith("layer_range:"):
            continue

        payload = line.split(":", 1)[1].strip()
        try:
            layer_range = ast.literal_eval(payload)
        except Exception as exc:
            raise SystemExit(
                f"Не удалось распознать layer_range для {current_model or 'неизвестной модели'}: {payload}"
            ) from exc

        if not isinstance(layer_range, (list, tuple)) or len(layer_range) < 2:
            raise SystemExit(
                f"Некорректный layer_range для {current_model or 'неизвестной модели'}: {payload}"
            )

        try:
            start = int(layer_range[0])
            stop = int(layer_range[1])
            step = int(layer_range[2]) if len(layer_range) >= 3 else 1
        except Exception as exc:
            raise SystemExit(
                f"Не удалось преобразовать значения layer_range для {current_model or 'неизвестной модели'}: {payload}"
            ) from exc

        if step == 0:
            raise SystemExit(
                f"Шаг не может быть равен 0 в layer_range для {current_model or 'неизвестной модели'}: {payload}"
            )

        forward = step > 0
        step = abs(step)
        delta = stop - start
        if (forward and delta < 0) or (not forward and delta > 0):
            raise SystemExit(
                f"Непоследовательные границы layer_range для {current_model or 'неизвестной модели'}: {payload}"
            )

        delta = abs(delta)
        exclusive = (delta + step - 1) // step
        inclusive = None
        if delta % step == 0:
            inclusive = exclusive + 1 if exclusive > 0 else 1

        candidates = []
        if exclusive > 0:
            candidates.append(exclusive)
        if inclusive is not None and inclusive != exclusive:
            candidates.append(inclusive)

        if not candidates:
            raise SystemExit(
                f"layer_range для {current_model or 'неизвестной модели'} не охватывает ни одного слоя: {payload}"
            )

        span = max(candidates)
        if span != expected:
            raise SystemExit(
                f"layer_range для {current_model or 'неизвестной модели'} описывает {span} слоёв (ожидалось {expected})."
            )

        normalised_model = _normalise_model(current_model)
        if not normalised_model:
            raise SystemExit("Не задан layer_range без имени модели")

        overrides.append(f"{normalised_model}={span}")

base_model_override = f"{_normalise_model(str(base_model_path))}={expected}"
if base_model_override not in overrides:
    overrides.append(base_model_override)

sys.stderr.write("Проверка layer_range прошла успешно.\n")
print(":".join(overrides))
PY
)"; then
    exit 1
  fi
  export MISTRAL_LAYER_RANGE_OVERRIDES="${overrides}"
}

verify_layer_ranges

echo "--- Профиль: DARE-Linear (локальный) ---"

[[ -z "${HF_TOKEN:-}" ]] && { echo "HF_TOKEN не задан"; exit 1; }

echo ">>> Загружаю исходные модели..."
huggingface-cli download Vikhrmodels/Vistral-24B-Instruct \
  --local-dir "${VISTRAL_DIR}" --exclude "*.bin"
huggingface-cli download TheDrummer/Cydonia-24B-v4.2.0 \
  --local-dir "${CYDONIA_DIR}"
huggingface-cli download mistralai/Mistral-Small-3.2-24B-Instruct-2506 \
  --local-dir "${BASE_MODEL_DIR}" --exclude "*.bin" --exclude "consolidated.safetensors"

echo ">>> Запускаю mergekit (dare_linear + профили)..."
mergekit-yaml "${CFG}" "${OUT}" \
  --cuda --copy-tokenizer --clone-tensors

echo "--- Готово ---"
echo "Результат сохранён в: ${OUT}"
