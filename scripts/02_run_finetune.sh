#!/bin/bash
set -e

echo "--- ФАЗА 2: ПОЛИРОВКА МОДЕЛИ С LoRA И CAME ---"

if [ ! -d "/workspace/merged_model" ]; then
    echo "Ошибка: Папка /workspace/merged_model не найдена. Запустите сначала скрипт Фазы 1."
    exit 1
fi

echo ">>> Запуск дообучения Axolotl (может занять >1 часа)..."
accelerate launch -m axolotl.cli.train ../configs/lora_came_tune.yml

echo "--- ФАЗА 2 ЗАВЕРШЕНА ---"
echo ">>> LoRA-адаптер готов в /workspace/merged_model_lora"