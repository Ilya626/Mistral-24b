#!/bin/bash
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
    echo "Этот скрипт необходимо запускать от имени root." >&2
    exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Не найден интерпретатор Python: ${PYTHON_BIN}" >&2
    exit 1
fi
PIP_CMD=("${PYTHON_BIN}" -m pip)

APT_PACKAGES=(
    build-essential
    git
    git-lfs
    python3-dev
)
MISSING_PACKAGES=()
for pkg in "${APT_PACKAGES[@]}"; do
    if ! dpkg -s "${pkg}" >/dev/null 2>&1; then
        MISSING_PACKAGES+=("${pkg}")
    fi
done
if ((${#MISSING_PACKAGES[@]})); then
    echo ">>> Обновление индексов пакетов..."
    if apt-get update; then
        echo ">>> Установка системных пакетов: ${MISSING_PACKAGES[*]}"
        apt-get install -y --no-install-recommends "${MISSING_PACKAGES[@]}"
        apt-get clean
        rm -rf /var/lib/apt/lists/*
    else
        echo "Предупреждение: не удалось обновить индексы apt." >&2
        echo "Установите отсутствующие пакеты вручную: ${MISSING_PACKAGES[*]}" >&2
        exit 1
    fi
else
    echo ">>> Все системные пакеты уже установлены, пропускаем apt-get install."
fi

if ! command -v git-lfs >/dev/null 2>&1; then
    echo "git-lfs не установлен корректно" >&2
    exit 1
fi

echo ">>> Инициализация Git LFS..."
git lfs install --skip-repo

echo ">>> Обновление pip..."
if ! "${PIP_CMD[@]}" install --upgrade --root-user-action=ignore pip; then
    echo "Предупреждение: не удалось обновить pip, продолжаем со штатной версией." >&2
fi

PYTHON_VERSION="$(${PYTHON_BIN} -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
TORCH_VERSION="${TORCH_VERSION:-2.6.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.21.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.6.0}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
if [[ "${PYTHON_VERSION}" != "3.11" ]]; then
    TORCH_VERSION="${TORCH_VERSION_FALLBACK:-2.6.0}"
    TORCHVISION_VERSION="${TORCHVISION_VERSION_FALLBACK:-0.21.0}"
    TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION_FALLBACK:-2.6.0}"
    TORCH_INDEX_URL="${TORCH_INDEX_URL_FALLBACK:-https://download.pytorch.org/whl/cu124}"
fi

if [[ "${SKIP_TORCH_INSTALL:-0}" == "1" ]]; then
    echo ">>> Пропускаем проверку и установку PyTorch по требованию (SKIP_TORCH_INSTALL=1)."
elif "${PYTHON_BIN}" - <<PYTHON >/dev/null 2>&1
import sys
try:
    import torch, torchvision, torchaudio
except ImportError:
    sys.exit(1)
expected = {
    "torch": "${TORCH_VERSION}",
    "torchvision": "${TORCHVISION_VERSION}",
    "torchaudio": "${TORCHAUDIO_VERSION}",
}
installed = {
    "torch": torch.__version__.split("+")[0],
    "torchvision": torchvision.__version__.split("+")[0],
    "torchaudio": torchaudio.__version__.split("+")[0],
}
for name, version in expected.items():
    if installed.get(name) != version:
        sys.exit(1)
sys.exit(0)
PYTHON
then
    echo ">>> PyTorch stack уже установлен: torch=${TORCH_VERSION}, torchvision=${TORCHVISION_VERSION}, torchaudio=${TORCHAUDIO_VERSION}"
else
    echo ">>> Настраиваем совместимый стек PyTorch..."
    "${PIP_CMD[@]}" uninstall -y torch torchvision torchaudio >/dev/null 2>&1 || true
    if ! "${PIP_CMD[@]}" install --no-cache-dir --root-user-action=ignore \
        --extra-index-url "${TORCH_INDEX_URL}" \
        "torch==${TORCH_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}" \
        "torchaudio==${TORCHAUDIO_VERSION}"; then
        echo "Ошибка: не удалось установить стек PyTorch (torch=${TORCH_VERSION})." >&2
        echo "Проверьте подключение к интернету или укажите альтернативный индекс через переменную TORCH_INDEX_URL." >&2
        exit 1
    fi
fi

if "${PIP_CMD[@]}" show modal >/dev/null 2>&1; then
    echo ">>> Удаление конфликтующего пакета modal..."
    "${PIP_CMD[@]}" uninstall -y modal
fi

echo ">>> Установка Python-зависимостей..."
PYTHON_PACKAGES=(
    "accelerate>=1.2.0"
    "datasets>=2.19.0"
    "huggingface_hub[cli]"
    "mergekit[hf]"
    "safetensors>=0.5.2"
    "sentencepiece"
    "transformers>=4.46.0"
    "tqdm"
)
"${PIP_CMD[@]}" install --upgrade --root-user-action=ignore "${PYTHON_PACKAGES[@]}"

echo ">>> Проверка ключевых Python-пакетов..."
"${PYTHON_BIN}" - <<'PYTHON'
import importlib
modules = [
    "torch",
    "torchvision",
    "torchaudio",
    "mergekit",
    "datasets",
    "transformers",
    "accelerate",
    "tqdm",
]
for module in modules:
    importlib.import_module(module)
print("Все модули успешно импортированы.")
PYTHON

echo ">>> Проверка доступности huggingface-cli..."
if ! huggingface-cli --help >/dev/null 2>&1; then
    echo "huggingface-cli недоступен" >&2
    exit 1
fi

echo ">>> Готово! Сетап завершен."
