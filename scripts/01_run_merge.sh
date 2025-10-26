#!/bin/bash
set -e # Выход при ошибке

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# --- НАСТРОЙКИ ---
# Путь к вашему новому конфигу
MERGE_CONFIG="${REPO_ROOT}/configs/slerp_merge.yml"
# Директории для моделей
VISTRAL_DIR="/workspace/Vistral-24B-Instruct"
CYDONIA_DIR="/workspace/Cydonia-24B-v4.2.0"
# Директория для итоговой модели
MERGED_DIR="/workspace/merged_model_slerp"

echo "--- УПРОЩЕННОЕ SLERP СЛИЯНИЕ МОДЕЛЕЙ ---"

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
