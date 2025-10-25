#!/bin/bash
# Этот скрипт устанавливает все необходимые зависимости
echo ">>> Обновление пакетов и установка git-lfs..."
apt-get update
apt-get install -y git git-lfs

echo ">>> Проверка установленной версии Python и PyTorch..."
if python -c "import importlib.util; import sys; sys.exit(0 if importlib.util.find_spec('torch') else 1)"; then
    INSTALLED_TORCH_VERSION="$(python -c 'import torch; print(torch.__version__)')"
    echo "PyTorch уже установлен (версия ${INSTALLED_TORCH_VERSION}). Пропускаем переустановку."
else
    PYTHON_VERSION="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    echo "PyTorch не найден. Текущая версия Python: ${PYTHON_VERSION}"

    TORCH_VERSION="2.1.2"
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"

    if [[ "${PYTHON_VERSION}" == "3.12" ]]; then
        TORCH_VERSION="2.4.0"
        TORCH_INDEX_URL="https://download.pytorch.org/whl/cu124"
    fi

    echo ">>> Установка PyTorch ${TORCH_VERSION} с поддержкой CUDA..."
    pip install --upgrade "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX_URL}"
fi

echo ">>> Установка mergekit..."
pip install --upgrade "mergekit[hf]"

echo ">>> Установка Axolotl..."
pip install --upgrade "axolotl[flash-attn,deepspeed]"

echo ">>> Установка huggingface_hub CLI..."
pip install --upgrade "huggingface_hub[cli]"

echo ">>> Проверка доступности huggingface-cli..."
if ! huggingface-cli --help >/dev/null 2>&1; then
    echo "Предупреждение: не удалось выполнить команду huggingface-cli --help"
    exit 1
fi

echo ">>> Инициализация Git LFS..."
git lfs install

echo ">>> Установка завершена!"
