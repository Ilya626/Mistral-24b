#!/bin/bash
set -euo pipefail

# Обновление пакетов и установка git-lfs
echo ">>> Обновление пакетов и установка git-lfs..."
apt-get update
apt-get install -y git git-lfs

# Проверка версии Python и установка PyTorch (актуальная версия для 2025, совместимая с axolotl и flash-attn)
PYTHON_VERSION="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
echo ">>> Проверка установленной версии Python (${PYTHON_VERSION}) и стека PyTorch..."

TORCH_VERSION="2.6.0"  # Актуально для совместимости с flash-attn, deepspeed и новыми функциями axolotl
TORCHVISION_VERSION="0.21.0"
TORCHAUDIO_VERSION="2.6.0"
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu124"

if [[ "${PYTHON_VERSION}" != "3.11" ]]; then  # Fallback для других версий
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

# Исправление повреждённой установки transformers (из предупреждения в логе)
echo ">>> Удаление повреждённой установки transformers..."
pip uninstall -y transformers tokenizers || true

# Установка axolotl с GitHub (свежая версия, чтобы избежать устаревших точных пинов с PyPI)
echo ">>> Установка Axolotl с GitHub и автоматическое разрешение зависимостей..."
pip install --upgrade --root-user-action=ignore git+https://github.com/axolotl-ai-cloud/axolotl.git#egg=axolotl[flash-attn,deepspeed]

# Установка mergekit с GitHub (свежая версия)
echo ">>> Установка mergekit без deps..."
pip install --upgrade --no-deps --root-user-action=ignore git+https://github.com/arcee-ai/mergekit.git#egg=mergekit[hf]

# Установка зависимостей mergekit (pip разрешит версии автоматически, совместимо с axolotl)
echo ">>> Установка зависимостей mergekit..."
MERGEKIT_DEPS=(
    "click"
    "immutables"
    "datasets"
    "peft"
    "protobuf"
    "pydantic"
    "scipy"
    "safetensors>=0.4.2"  # Обновлённый пин из актуальных требований mergekit
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
