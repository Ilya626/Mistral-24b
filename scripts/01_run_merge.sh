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

# Запускаем слияние
mergekit-yaml "${MERGE_CONFIG}" "${MERGED_DIR}" --cuda --copy-tokenizer --allow-crimes

echo "--- СЛИЯНИЕ ЗАВЕРШЕНО ---"
echo ">>> Упрощенная слитая модель готова в ${MERGED_DIR}"
