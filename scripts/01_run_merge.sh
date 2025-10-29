#!/bin/bash
set -e # Выход при ошибке

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Ensure our local Python patches are active for mergekit invocations.
export PYTHONPATH="${REPO_ROOT}/pythonpath${PYTHONPATH:+:${PYTHONPATH}}"

# --- НАСТРОЙКИ ---
# Путь к конфигу мержа можно переопределить аргументом --profile или переменной MERGE_CONFIG
DEFAULT_CONFIG="${REPO_ROOT}/configs/slerp_merge.yml"
MERGE_CONFIG="${MERGE_CONFIG:-${DEFAULT_CONFIG}}"
# Директории для моделей
VISTRAL_DIR="/workspace/Vistral-24B-Instruct"
CYDONIA_DIR="/workspace/Cydonia-24B-v4.2.0"
# Директория для итоговой модели
MERGED_DIR="/workspace/merged_model"

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
      local target="${src}/${rel}"
      if [[ -f "${target}" && ! -f "${dest}/${rel}" ]]; then
        echo "    ↳ Копирую ${rel} из ${src}"
        cp "${target}" "${dest}/${rel}"
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
        "проверьте, что инструменты квантования умеют работать только с tokenizer.json.\n"
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

usage() {
  cat <<EOF
Использование: $0 [--profile /path/to/profile.yml]

Опции:
  --profile PATH   Явно указать YAML-профиль для mergekit-yaml (по умолчанию configs/slerp_merge.yml
                    или значение переменной MERGE_CONFIG).
  -h, --help       Показать эту справку и выйти.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      if [[ -z "$2" ]]; then
        echo "Ошибка: флаг --profile требует путь к YAML-файлу." >&2
        exit 1
      fi
      MERGE_CONFIG="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Неизвестный аргумент: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -f "$MERGE_CONFIG" ]]; then
  echo "Ошибка: не найден профиль мержа '$MERGE_CONFIG'." >&2
  exit 1
fi

echo "--- УПРОЩЕННОЕ SLERP СЛИЯНИЕ МОДЕЛЕЙ ---"
echo "Используем профиль: ${MERGE_CONFIG}"

# Проверка переменной окружения
if [ -z "$HF_TOKEN" ]; then
  echo "Ошибка: Переменная окружения HF_TOKEN не установлена."
  exit 1
fi

echo ">>> Шаг 1/2: Загрузка 2 моделей..."
huggingface-cli download Vikhrmodels/Vistral-24B-Instruct --local-dir "${VISTRAL_DIR}" --exclude "*.bin"
huggingface-cli download TheDrummer/Cydonia-24B-v4.2.0 --local-dir "${CYDONIA_DIR}"

echo ">>> Шаг 2/2: Запуск слияния SLERP..."
# Создаем директорию для итоговой модели, если ее нет
mkdir -p "${MERGED_DIR}"

# Запускаем слияние. Не используем --copy-tokenizer, чтобы mergekit собрал union
# токенизаторов (Vistral + Cydonia) согласно профилю.
mergekit-yaml "${MERGE_CONFIG}" "${MERGED_DIR}" --cuda --allow-crimes

echo ">>> Синхронизирую служебные файлы токенайзера..."
copy_tokenizer_sidecars "${MERGED_DIR}" "${VISTRAL_DIR}" "${CYDONIA_DIR}"
validate_union_tokenizer "${MERGED_DIR}" "${VISTRAL_DIR}" "${CYDONIA_DIR}"

echo "--- СЛИЯНИЕ ЗАВЕРШЕНО ---"
echo ">>> Упрощенная слитая модель готова в ${MERGED_DIR}"
