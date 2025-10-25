#!/bin/bash
# Этот скрипт устанавливает все необходимые зависимости
echo ">>> Обновление пакетов и установка git-lfs..."
apt-get update
apt-get install -y git git-lfs

echo ">>> Установка PyTorch с поддержкой CUDA..."
pip install --upgrade torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121

echo ">>> Установка mergekit..."
pip install --upgrade "mergekit[hf]"

echo ">>> Установка Axolotl..."
pip install --upgrade "axolotl[flash-attn,deepspeed]"

echo ">>> Установка huggingface_hub CLI..."
pip install --upgrade "huggingface_hub[cli]"

echo ">>> Проверка доступности huggingface-cli..."
huggingface-cli --version

echo ">>> Инициализация Git LFS..."
git lfs install

echo ">>> Установка завершена!"
