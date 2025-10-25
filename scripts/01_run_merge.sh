#!/bin/bash
set -e # Выход при ошибке

echo "--- ФАЗА 1: ГИБРИДНОЕ СЛИЯНИЕ МОДЕЛЕЙ ---"

# Проверка переменных окружения
if [ -z "$HF_TOKEN" ]; then
    echo "Ошибка: Переменная окружения HF_TOKEN не установлена."
    exit 1
fi

echo ">>> Шаг 1/2: Загрузка 3 моделей (может занять >30 минут)..."
huggingface-cli download Vikhrmodels/Vistral-24B-Instruct --local-dir /workspace/Vistral-24B-Instruct --exclude "*.bin"
huggingface-cli download TheDrummer/Cydonia-24B-v3.1 --local-dir /workspace/Cydonia-24B-v3.1 --exclude "*.bin"
huggingface-cli download mistralai/Mistral-Small-3.1-24B-Base-2503 --local-dir /workspace/Mistral-Small-3.1-24B-Base-2503 --exclude "*.bin"

echo ">>> Шаг 2/2: Запуск слияния DARE TIES Hybrid..."
mergekit-yaml ../configs/dare_hybrid_merge.yml /workspace/merged_model --cuda --copy-tokenizer --allow-crimes

echo "--- ФАЗА 1 ЗАВЕРШЕНА ---"
echo ">>> 'Сырая' слитая модель готова в /workspace/merged_model"