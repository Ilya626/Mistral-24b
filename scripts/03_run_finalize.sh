#!/bin/bash
set -e

echo "--- ФАЗА 3: ФИНАЛИЗАЦИЯ И ЗАГРУЗКА НА HUGGING FACE ---"

# Проверка переменных окружения
if [ -z "$HF_TOKEN" ] || [ -z "$HF_USERNAME" ] || [ -z "$NEW_MODEL_NAME" ]; then
    echo "Ошибка: Убедитесь, что переменные HF_TOKEN, HF_USERNAME и NEW_MODEL_NAME установлены."
    exit 1
fi

if [ ! -d "/workspace/merged_model_lora" ]; then
    echo "Ошибка: Папка /workspace/merged_model_lora не найдена. Запустите сначала скрипт Фазы 2."
    exit 1
fi

echo ">>> Шаг 1/3: Впекание LoRA-адаптера в модель..."
mergekit-hf merge /workspace/merged_model /workspace/merged_model_lora /workspace/final_model --cuda

echo ">>> Шаг 2/3: Подготовка и загрузка на Hugging Face Hub..."
huggingface-cli repo create ${NEW_MODEL_NAME} --type model --exist-ok

# Используем git clone & cp вместо `upload` для лучшей обработки больших файлов
git clone https://${HF_USERNAME}:${HF_TOKEN}@huggingface.co/${HF_USERNAME}/${NEW_MODEL_NAME} /workspace/hf_repo
cp /workspace/final_model/* /workspace/hf_repo/

echo ">>> Шаг 3/3: Отправка файлов (может быть долго)..."
cd /workspace/hf_repo
git lfs track "*.safetensors"
git add .
git commit -m "Final model from automated pipeline"
git push

echo "--- КОНВЕЙЕР УСПЕШНО ЗАВЕРШЕН! ---"
echo ">>> Ваша модель доступна в репозитории: https://huggingface.co/${HF_USERNAME}/${NEW_MODEL_NAME}"
echo ">>> !!! НЕ ЗАБУДЬТЕ ОСТАНОВИТЬ ПОД НА RUNPOD !!!"