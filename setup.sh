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
    TORCH_VERSION="2.4.0"  # Изменено на 2.4.0 для совпадения с RunPod-образом
    TORCHVISION_VERSION="0.19.0"
    TORCHAUDIO_VERSION="2.4.0"
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
    pip install --no-cache-dir \
        --index-url "${TORCH_INDEX_URL}" \
        "torch==${TORCH_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}" \
        "torchaudio==${TORCH_AUDIO_VERSION}"
fi

echo ">>> Установка mergekit без автоматической установки зависимостей..."
# mergekit 0.0.6 жестко требует accelerate~=1.3.0, что совпадает с требованиями
# инструмента mergekit-yaml. Чтобы предотвратить конфликты, управляем версиями вручную.
pip install --upgrade --no-deps "mergekit[hf]"

echo ">>> Установка зависимостей mergekit с фиксированными версиями..."
# Устанавливаем все обязательные зависимости mergekit, кроме accelerate. Так мы
# удовлетворяем требованиям mergekit (click, immutables, safetensors~=0.5.2 и т.д.) и
# одновременно сохраняем актуальную версию accelerate, необходимую Axolotl.
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
pip install --upgrade "${MERGEKIT_DEPS[@]}"

echo ">>> Установка Axolotl..."
pip install --upgrade "axolotl[flash-attn,deepspeed]"

echo ">>> Фиксация accelerate в диапазоне, совместимом с mergekit..."
pip install --upgrade "accelerate>=0.27.0,<0.29.0"  # Совместимо с axolotl и mergekit

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
