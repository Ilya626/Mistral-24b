#!/bin/bash
# Этот скрипт устанавливает все необходимые зависимости

echo ">>> Обновление пакетов и установка git-lfs..."
apt-get update
apt-get install -y git git-lfs

echo ">>> Установка Python-библиотек..."
pip install -U "mergekit[hf]" "axolotl[flash-attn,deepspeed]" "huggingface_hub[cli]"

echo ">>> Инициализация Git LFS..."
git lfs install

echo ">>> Установка завершена!"