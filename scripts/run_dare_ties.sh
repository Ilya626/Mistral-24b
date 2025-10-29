#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Ensure our Python patches are picked up before invoking mergekit.
export PYTHONPATH="${REPO_ROOT}/pythonpath${PYTHONPATH:+:${PYTHONPATH}}"

CFG="${CFG:-${REPO_ROOT}/configs/dare_hybrid_merge.yml}"
OUT="${OUT:-/workspace/merged_dare_ties}"

VISTRAL_DIR="${VISTRAL_DIR:-/workspace/Vistral-24B-Instruct}"
CYDONIA_DIR="${CYDONIA_DIR:-/workspace/Cydonia-24B-v4.2.0}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-/workspace/Mistral-Small-3.2-24B-Instruct-2506}"
OFFICIAL_TOKENIZER_REPO="${OFFICIAL_TOKENIZER_REPO:-mistralai/Mistral-Small-3.1-24B-Base-2503}"

copy_tokenizer_sidecars() {
  local dest="$1"
  shift
  local files=(
    "tokenizer_config.json"
    "special_tokens_map.json"
    "tekken.json"
    "preprocessor_config.json"
    "processor_config.json"
  )
  for src in "$@"; do
    [[ -d "${src}" ]] || continue
    for rel in "${files[@]}"; do
      local candidate="${src}/${rel}"
      if [[ -f "${candidate}" && ! -f "${dest}/${rel}" ]]; then
        echo "    ↳ Копирую ${rel} из ${src}"
        cp "${candidate}" "${dest}/${rel}"
      fi
    done
  done
}

validate_union_tokenizer() {
  local dest="$1"
  shift
  python3 - "$dest" "$@" <<'PY'
import json
import sys
from pathlib import Path


def _load_token_sets(root: Path) -> tuple[set[str], set[str]]:
    vocab: set[str] = set()
    specials: set[str] = set()

    def _consume(obj: object) -> None:
        if isinstance(obj, str):
            specials.add(obj)
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, str):
                    specials.add(item)
        elif isinstance(obj, dict):
            value = obj.get("content") or obj.get("token")
            if isinstance(value, str):
                specials.add(value)

    tok_json = root / "tokenizer.json"
    if tok_json.exists():
        try:
            with tok_json.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception as exc:  # pragma: no cover - safety guard
            raise SystemExit(f"Ошибка: не удалось прочитать {tok_json}: {exc}") from exc

        model = data.get("model")
        if isinstance(model, dict):
            vocab_map = model.get("vocab")
            if isinstance(vocab_map, dict):
                vocab.update(k for k in vocab_map.keys() if isinstance(k, str))

        for token in data.get("added_tokens", []):
            if not isinstance(token, dict):
                continue
            content = token.get("content")
            if isinstance(content, str):
                vocab.add(content)
                if token.get("special", False):
                    specials.add(content)

    config = root / "tokenizer_config.json"
    if config.exists():
        try:
            with config.open("r", encoding="utf-8") as handle:
                cfg = json.load(handle)
        except Exception as exc:  # pragma: no cover
            raise SystemExit(f"Ошибка: не удалось прочитать {config}: {exc}") from exc

        special_map = cfg.get("special_tokens_map")
        if isinstance(special_map, dict):
            for value in special_map.values():
                _consume(value)

        for key in (
            "bos_token",
            "eos_token",
            "unk_token",
            "pad_token",
            "cls_token",
            "sep_token",
            "mask_token",
        ):
            _consume(cfg.get(key))

    special_json = root / "special_tokens_map.json"
    if special_json.exists():
        try:
            with special_json.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception as exc:  # pragma: no cover
            raise SystemExit(f"Ошибка: не удалось прочитать {special_json}: {exc}") from exc

        if isinstance(data, dict):
            for value in data.values():
                _consume(value)

    return vocab, specials


dest = Path(sys.argv[1])
sources = [Path(p) for p in sys.argv[2:] if Path(p).is_dir()]

tokenizer_json = dest / "tokenizer.json"
if not tokenizer_json.exists():
    raise SystemExit(
        f"Ошибка: {tokenizer_json} не найден. Mergekit должен сгенерировать объединённый tokenizer.json."
    )

spm = dest / "tokenizer.model"
if not spm.exists():
    sys.stderr.write(
        "[!] Предупреждение: tokenizer.model отсутствует в целевой директории — "
        "убедитесь, что инструменты квантования могут работать с tokenizer.json.\n"
    )

dest_vocab, dest_special = _load_token_sets(dest)
if not dest_vocab:
    raise SystemExit(
        "Ошибка: не удалось извлечь словарь из объединённого токенайзера. Проверьте tokenizer.json."
    )

problems = False
for src in sources:
    try:
        src_vocab, src_special = _load_token_sets(src)
    except SystemExit as exc:
        sys.stderr.write(f"[!] Предупреждение: пропускаю проверку {src}: {exc}\n")
        continue

    if not src_vocab:
        sys.stderr.write(
            f"[!] Предупреждение: не удалось извлечь словарь из {src}/tokenizer.json — пропускаю проверку.\n"
        )
        continue

    missing = [token for token in src_vocab if token not in dest_vocab]
    if missing:
        problems = True
        sample = ", ".join(missing[:5])
        sys.stderr.write(
            f"Ошибка: объединённый токенайзер не содержит {len(missing)} токенов из {src}. "
            f"Примеры: {sample}\n"
        )

    special_missing = [token for token in src_special if token and token not in dest_special]
    if special_missing:
        problems = True
        sys.stderr.write(
            f"Ошибка: отсутствуют специальные токены из {src}: "
            + ", ".join(special_missing[:5])
            + "\n"
        )

if problems:
    raise SystemExit(1)

print("✓ Объединённый токенайзер содержит словари всех исходных моделей.")
PY
}

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

require_files() {
  local label="$1"
  local root="$2"
  local repo="$3"
  shift 3
  local missing=0
  for rel in "$@"; do
    local target="${root}/${rel}"
    if [[ ! -f "${target}" ]]; then
      echo "Ошибка: для ${label} не найден ${rel} (ожидался ${target})." >&2
      missing=1
    fi
  done
  if (( missing )); then
    echo "Подсказка: повторите загрузку: huggingface-cli download ${repo} --local-dir \"${root}\" --force-download" >&2
    exit 1
  fi
  echo "✓ ${label}: обязательные файлы на месте."
}

require_any_file() {
  local label="$1"
  local root="$2"
  local repo="$3"
  shift 3
  for rel in "$@"; do
    if [[ -f "${root}/${rel}" ]]; then
      echo "✓ ${label}: найден ${rel}."
      return 0
    fi
  done
  echo "Ошибка: для ${label} не найден ни один из файлов: $* (каталог ${root})." >&2
  echo "Подсказка: повторите загрузку: huggingface-cli download ${repo} --local-dir \"${root}\" --force-download" >&2
  exit 1
}

sync_base_tokenizer() {
  if ls "${BASE_MODEL_DIR}"/tokenizer.* >/dev/null 2>&1; then
    return 0
  fi

  echo "[!] Базовая модель без токенайзера. Скачиваю из ${OFFICIAL_TOKENIZER_REPO}."
  local files=(
    tokenizer.json
    tokenizer_config.json
    special_tokens_map.json
    tekken.json
    preprocessor_config.json
    processor_config.json
  )
  for rel in "${files[@]}"; do
    huggingface-cli download "${OFFICIAL_TOKENIZER_REPO}" \
      --local-dir "${BASE_MODEL_DIR}" \
      --include "${rel}" \
      --force-download \
      --local-dir-use-symlinks False >/dev/null
  done
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

echo "--- Профиль: DARE-TIES (гибрид) ---"

[[ -z "${HF_TOKEN:-}" ]] && { echo "HF_TOKEN не задан"; exit 1; }

echo ">>> Загружаю исходные модели..."
huggingface-cli download Vikhrmodels/Vistral-24B-Instruct \
  --local-dir "${VISTRAL_DIR}" --exclude "*.bin"
huggingface-cli download TheDrummer/Cydonia-24B-v4.2.0 \
  --local-dir "${CYDONIA_DIR}"
huggingface-cli download mistralai/Mistral-Small-3.2-24B-Instruct-2506 \
  --local-dir "${BASE_MODEL_DIR}" --exclude "*.bin" --exclude "consolidated.safetensors"

sync_base_tokenizer

require_files "Vistral" "${VISTRAL_DIR}" "Vikhrmodels/Vistral-24B-Instruct" \
  "config.json"
require_any_file "Vistral" "${VISTRAL_DIR}" "Vikhrmodels/Vistral-24B-Instruct" \
  "tokenizer.model" "tokenizer.json"
require_files "Cydonia" "${CYDONIA_DIR}" "TheDrummer/Cydonia-24B-v4.2.0" \
  "config.json"
require_any_file "Cydonia" "${CYDONIA_DIR}" "TheDrummer/Cydonia-24B-v4.2.0" \
  "tokenizer.model" "tokenizer.json"
require_files "базовой модели" "${BASE_MODEL_DIR}" "mistralai/Mistral-Small-3.2-24B-Instruct-2506" \
  "config.json"
require_any_file "базовой модели" "${BASE_MODEL_DIR}" "mistralai/Mistral-Small-3.2-24B-Instruct-2506" \
  "tokenizer.model" "tokenizer.json"

echo ">>> Запускаю mergekit (dare_ties + slerp)..."
# Не передаём --copy-tokenizer, чтобы mergekit сформировал объединённый словарь
# Vistral + Cydonia, как указано в конфиге.
mergekit-yaml "${CFG}" "${OUT}" \
  --cuda --clone-tensors

echo ">>> Синхронизирую служебные файлы токенайзера..."
copy_tokenizer_sidecars "${OUT}" "${VISTRAL_DIR}" "${CYDONIA_DIR}" "${BASE_MODEL_DIR}"
validate_union_tokenizer "${OUT}" "${VISTRAL_DIR}" "${CYDONIA_DIR}" "${BASE_MODEL_DIR}"

echo "--- Готово ---"
echo "Результат сохранён в: ${OUT}"
