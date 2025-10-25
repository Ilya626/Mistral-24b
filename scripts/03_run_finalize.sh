#!/bin/bash
set -e

echo "--- ФАЗА 3: ФИНАЛИЗАЦИЯ И ЗАГРУЗКА НА HUGGING FACE ---"

# Проверка переменных окружения
if [ -z "$HF_TOKEN" ] || [ -z "$HF_USERNAME" ] || [ -z "$NEW_MODEL_NAME" ] || [ -z "$HF_EMAIL" ]; then
    echo "Ошибка: Убедитесь, что переменные HF_TOKEN, HF_USERNAME, HF_EMAIL и NEW_MODEL_NAME установлены."
    exit 1
fi

if [ ! -d "/workspace/merged_model_lora" ]; then
    echo "Ошибка: Папка /workspace/merged_model_lora не найдена. Запустите сначала скрипт Фазы 2."
    exit 1
fi

echo ">>> Очистка предыдущих артефактов финальной модели..."
rm -rf /workspace/final_model

cd /workspace

echo ">>> Шаг 1/3: Впекание LoRA-адаптера в модель..."
mergekit-hf merge /workspace/merged_model /workspace/merged_model_lora /workspace/final_model --cuda

echo ">>> Шаг 2/3: Подготовка и загрузка на Hugging Face Hub..."
huggingface-cli repo create ${NEW_MODEL_NAME} --type model --exist-ok
rm -rf /workspace/hf_repo
git clone https://${HF_USERNAME}:${HF_TOKEN}@huggingface.co/${HF_USERNAME}/${NEW_MODEL_NAME} /workspace/hf_repo
cp -r /workspace/final_model/* /workspace/hf_repo/
echo ">>> Артефакты успешно скопированы в локальный репозиторий."

echo ">>> Быстрая проверка модели на сэмплах grandmaster2..."
python3 /workspace/Mistral-24b/scripts/evaluate_prompts.py \
    --model /workspace/final_model \
    --dataset grandmaster2 \
    --dataset-split "train[:8]" \
    --max-samples 8 \
    --output-dir /workspace/evaluation_logs || \
    echo ">>> Предупреждение: оценка не выполнена (см. сообщение выше)."

echo ">>> Шаг 3/3: Отправка файлов (может быть долго)..."
cd /workspace/hf_repo
git config user.name "${HF_USERNAME}"
git config user.email "${HF_EMAIL}"
git lfs track "*.safetensors"
git add .

if git diff --cached --quiet; then
    echo ">>> Нет изменений для коммита. Пропускаем отправку."
else
    git commit -m "Final model from automated pipeline"
    git push
fi

echo "--- КОНВЕЙЕР УСПЕШНО ЗАВЕРШЕН! ---"
echo ">>> Ваша модель доступна в репозитории: https://huggingface.co/${HF_USERNAME}/${NEW_MODEL_NAME}"
echo ">>> Фаза 3 успешно завершена."
echo ">>> !!! НЕ ЗАБУДЬТЕ ОСТАНОВИТЬ ПОД НА RUNPOD !!!"
