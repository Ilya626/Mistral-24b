#!/bin/bash
set -euo pipefail

# Этот скрипт устанавливает все необходимые зависимости
echo ">>> Обновление пакетов и установка git-lfs..."
apt-get update
apt-get install -y git git-lfs

PYTHON_VERSION="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
echo ">>> Проверка установленной версии Python (${PYTHON_VERSION}) и стека PyTorch..."

TORCH_VERSION="2.1.2"
TORCHVISION_VERSION="0.16.2"
TORCHAUDIO_VERSION="2.1.2"
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"

if [[ "${PYTHON_VERSION}" == "3.12" ]] || [[ "${PYTHON_VERSION}" == "3.11" ]] || [[ "${PYTHON_VERSION}" == "3.10" ]]; then
    TORCH_VERSION="2.6.0"  # Обновлено для совместимости с axolotl и flash-attn
    TORCHVISION_VERSION="0.21.0"
    TORCHAUDIO_VERSION="2.6.0"
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cu124"
fi

if python - <<PYTHON
import sys

required = {
    "torch": "${TORCH_VERSION}",
    "torchvision": "${TORCHVISION_VERSION}",
    "torchaudio": "${TORCHAUDIO_VERSION}",
}

try:
    import torch, torchvision, torchaudio  # noqa: F401
except ImportError:
    sys.exit(1)

installed = {
    "torch": torch.__version__.split("+")[0],
    "torchvision": torchvision.__version__.split("+")[0],
    "torchaudio": torchaudio.__version__.split("+")[0],
}

for name, required_version in required.items():
    if installed[name] != required_version:
        sys.exit(1)
sys.exit(0)
PYTHON
then
    echo "PyTorch stack уже выровнен: torch=${TORCH_VERSION}, torchvision=${TORCHVISION_VERSION}, torchaudio=${TORCHAUDIO_VERSION}"
else
    echo ">>> Обнаружены несовместимые версии PyTorch. Переустанавливаем совместимый стек..."
    pip uninstall -y torch torchvision torchaudio || true
    pip install --no-cache-dir --root-user-action=ignore \
        --index-url "${TORCH_INDEX_URL}" \
        "torch==${TORCH_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}" \
        "torchaudio==${TORCHAUDIO_VERSION}"
fi

echo ">>> Установка Axolotl (первым, чтобы он потянул правильные зависимости)..."
pip install --upgrade --root-user-action=ignore "axolotl[flash-attn,deepspeed]"

echo ">>> Установка mergekit с GitHub (свежая версия, без устаревших зависимостей)..."
pip install --upgrade --no-deps --root-user-action=ignore git+https://github.com/arcee-ai/mergekit.git#egg=mergekit[hf]

echo ">>> Установка зависимостей mergekit с фиксированными версиями..."
# Устанавливаем зависимости mergekit, кроме accelerate (пусть axolotl управляет версией)
MERGEKIT_DEPS=(
    "click==8.1.8"
    "immutables==0.20"
    "datasets"
    "peft"
    "protobuf"
    "pydantic>=2.10.6,<2.11"
    "scipy"
    "safetensors>=0.5.2,<0.6.0"
    "transformers"
    "sentencepiece"
    "hf_transfer"
    "einops"
)
pip install --upgrade --root-user-action=ignore "${MERGEKIT_DEPS[@]}"

# Удалена фиксация accelerate — pip разрешит конфликты автоматически

echo ">>> Установка huggingface_hub CLI..."
pip install --upgrade --root-user-action=ignore "huggingface_hub[cli]"

echo ">>> Проверка доступности huggingface-cli..."
if ! huggingface-cli --help >/dev/null 2>&1; then
    echo "Предупреждение: не удалось выполнить команду huggingface-cli --help"
    exit 1
fi

echo ">>> Инициализация Git LFS..."
git lfs install

echo ">>> Установка завершена!"
