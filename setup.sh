#!/bin/bash
set -euo pipefail

# Обновление пакетов и установка git-lfs
echo ">>> Обновление пакетов и установка git-lfs..."
apt-get update
apt-get install -y git git-lfs

# Проверка версии Python и установка PyTorch (совместимо с RunPod-образом, но обновлено для axolotl)
PYTHON_VERSION="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
echo ">>> Проверка установленной версии Python (${PYTHON_VERSION}) и стека PyTorch..."

TORCH_VERSION="2.6.0"  # Актуально для 2025, совместимо с flash-attn и deepspeed
TORCHVISION_VERSION="0.21.0"
TORCHAUDIO_VERSION="2.6.0"
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu124"

if [[ "${PYTHON_VERSION}" != "3.11" ]]; then  # Для других версий fallback
    TORCH_VERSION="2.1.2"
    TORCHVISION_VERSION="0.16.2"
    TORCHAUDIO_VERSION="2.1.2"
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"
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

# Установка axolotl первым — пусть потянет свои зависимости (accelerate, peft, transformers и т.д.)
echo ">>> Установка Axolotl и автоматическое разрешение зависимостей..."
pip install --upgrade --root-user-action=ignore "axolotl[flash-attn,deepspeed]"

# Установка mergekit с GitHub (свежая версия) без deps, затем его зависимости — pip разрешит
echo ">>> Установка mergekit без deps..."
pip install --upgrade --no-deps --root-user-action=ignore git+https://github.com/arcee-ai/mergekit.git#egg=mergekit[hf]

echo ">>> Установка зависимостей mergekit (pip разрешит версии автоматически)..."
MERGEKIT_DEPS=(
    "click"
    "immutables"
    "datasets"
    "peft"
    "protobuf"
    # mergekit пока не совместим с Pydantic v2 (см. ошибка schema-for-unknown-type)
    "pydantic<2"
    "scipy"
    "safetensors>=0.5.2,<0.6.0"  # Единственный жёсткий пин, чтобы не сломать
    "transformers"
    "sentencepiece"
    "hf_transfer"
    "einops"
)
pip install --upgrade --root-user-action=ignore "${MERGEKIT_DEPS[@]}"

# Установка huggingface_hub CLI
echo ">>> Установка huggingface_hub CLI..."
pip install --upgrade --root-user-action=ignore "huggingface_hub[cli]"

# Проверка huggingface-cli
echo ">>> Проверка доступности huggingface-cli..."
if ! huggingface-cli --help >/dev/null 2>&1; then
    echo "Предупреждение: не удалось выполнить команду huggingface-cli --help"
    exit 1
fi

# Инициализация Git LFS
echo ">>> Инициализация Git LFS..."
git lfs install

echo ">>> Установка завершена!"
