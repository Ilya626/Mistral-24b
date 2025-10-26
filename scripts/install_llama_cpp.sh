#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
THIRD_PARTY_DIR="${REPO_ROOT}/third_party"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-${THIRD_PARTY_DIR}/llama.cpp}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Python interpreter not found: ${PYTHON_BIN}" >&2
    exit 1
fi

APT_PACKAGES=(
    build-essential
    cmake
    git
    ninja-build
    python3-dev
)

MISSING=()
for pkg in "${APT_PACKAGES[@]}"; do
    if ! dpkg -s "${pkg}" >/dev/null 2>&1; then
        MISSING+=("${pkg}")
    fi
done

if ((${#MISSING[@]})); then
    echo ">>> Installing missing system packages: ${MISSING[*]}"
    apt-get update
    apt-get install -y --no-install-recommends "${MISSING[@]}"
    apt-get clean
    rm -rf /var/lib/apt/lists/*
fi

mkdir -p "${THIRD_PARTY_DIR}"

if [[ -d "${LLAMA_CPP_DIR}/.git" ]]; then
    echo ">>> Updating existing llama.cpp checkout at ${LLAMA_CPP_DIR}"
    git -C "${LLAMA_CPP_DIR}" pull --ff-only
else
    echo ">>> Cloning llama.cpp into ${LLAMA_CPP_DIR}"
    git clone --depth 1 https://github.com/ggerganov/llama.cpp "${LLAMA_CPP_DIR}"
fi

echo ">>> Building llama.cpp quantization tools"
cmake -S "${LLAMA_CPP_DIR}" -B "${LLAMA_CPP_DIR}/build" -G Ninja -D CMAKE_BUILD_TYPE=Release
cmake --build "${LLAMA_CPP_DIR}/build" --target llama-quantize

PIP_CMD=("${PYTHON_BIN}" -m pip install --upgrade --root-user-action=ignore)
"${PIP_CMD[@]}" pip

PYTHON_PACKAGES=(
    "llama-cpp-python>=0.2.90"
    "huggingface_hub[cli]"
    "hf-transfer"
    "transformers>=4.36"
    "accelerate"
    "sentencepiece"
    "safetensors"
    "numpy"
    "tqdm"
)

"${PYTHON_BIN}" -m pip install --upgrade --root-user-action=ignore "${PYTHON_PACKAGES[@]}"

echo ">>> llama.cpp environment ready"
