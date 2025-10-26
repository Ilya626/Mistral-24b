#!/bin/bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/run_quantize.sh MODEL_SOURCE [OUTPUT_DIR]

MODEL_SOURCE  Local path, Hugging Face repo ID, or https://huggingface.co URL for the model.
OUTPUT_DIR    Optional directory for GGUF artifacts (defaults vary, see README).
EOF
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
    usage >&2
    exit 1
fi

MODEL_SOURCE="${1}"
USER_OUTPUT_DIR="${2:-}"

UPLOAD_REPO="${HF_UPLOAD_REPO:-}"
UPLOAD_INCLUDE_FLOAT="${HF_UPLOAD_INCLUDE_FLOAT:-0}"
UPLOAD_PRIVATE="${HF_UPLOAD_PRIVATE:-1}"
UPLOAD_PREFIX="${HF_UPLOAD_PREFIX:-gguf}"
UPLOAD_MESSAGE="${HF_UPLOAD_MESSAGE:-Upload quantized GGUF weights}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-${REPO_ROOT}/third_party/llama.cpp}"

TEMP_MODEL_DIR=""
CACHED_SOURCE_REPO_ID=""
cleanup() {
    if [[ -n "${TEMP_MODEL_DIR}" && -d "${TEMP_MODEL_DIR}" ]]; then
        rm -rf "${TEMP_MODEL_DIR}"
    fi
}
trap cleanup EXIT

normalize_repo_id() {
    local source="$1"
    source="${source#hf://}"
    source="${source#https://huggingface.co/}"
    source="${source#http://huggingface.co/}"
    source="${source%%/}"
    source="${source%%\?*}"
    source="${source%%#*}"
    if [[ "${source}" == *"/resolve/"* ]]; then
        source="${source%%/resolve/*}"
    fi
    echo "${source}"
}

CACHED_SOURCE_REPO_ID="$(normalize_repo_id "${MODEL_SOURCE}")"

ensure_hf_cli() {
    if ! command -v huggingface-cli >/dev/null 2>&1; then
        echo "huggingface-cli not found. Run scripts/install_llama_cpp.sh to install dependencies." >&2
        exit 1
    fi
}

ensure_hf_login() {
    if ! huggingface-cli whoami >/dev/null 2>&1; then
        echo "huggingface-cli login не выполнен. Запустите 'huggingface-cli login' или задайте переменную HF_TOKEN." >&2
        exit 1
    fi
}

ensure_hf_repo() {
    local repo_id="$1"
    local -a create_args=(repo create "${repo_id}" --type model)
    if [[ "${UPLOAD_PRIVATE}" == "1" ]]; then
        create_args+=(--private)
    fi
    if ! huggingface-cli "${create_args[@]}" >/dev/null 2>&1; then
        echo ">>> Hugging Face repo ${repo_id} already exists или создать не удалось. Продолжаем с загрузкой." >&2
    else
        echo ">>> Создан новый репозиторий Hugging Face: ${repo_id}"
    fi
}

upload_file_to_hf() {
    local repo_id="$1"
    local local_path="$2"
    local remote_path="$3"

    huggingface-cli upload "${repo_id}" "${local_path}" "${remote_path}" \
        --repo-type model \
        --commit-message "${UPLOAD_MESSAGE}" >/dev/null
    echo ">>> Загружено ${local_path} → https://huggingface.co/${repo_id}/resolve/main/${remote_path}"
}

resolve_model_dir() {
    local source="$1"
    if [[ -d "${source}" ]]; then
        echo "${source}"
        return 0
    fi

    if ! command -v huggingface-cli >/dev/null 2>&1; then
        echo "huggingface-cli not found. Run scripts/install_llama_cpp.sh to install dependencies." >&2
        return 1
    fi

    local repo_id
    repo_id="$(normalize_repo_id "${source}")"
    if [[ -z "${repo_id}" || "${repo_id}" == */*/* ]]; then
        echo "Unable to infer Hugging Face repo id from source: ${source}" >&2
        return 1
    fi

    CACHED_SOURCE_REPO_ID="${repo_id}"

    TEMP_MODEL_DIR="$(mktemp -d)"
    if ! huggingface-cli whoami >/dev/null 2>&1; then
        echo ">>> huggingface-cli login не выполнен. Для приватных моделей авторизуйтесь или задайте переменную HF_TOKEN." >&2
    fi

    echo ">>> Downloading Hugging Face model ${repo_id}" >&2
    huggingface-cli download "${repo_id}" \
        --repo-type model \
        --local-dir "${TEMP_MODEL_DIR}" \
        --local-dir-use-symlinks False

    printf '%s\n' "${TEMP_MODEL_DIR}"
}

MODEL_DIR="$(resolve_model_dir "${MODEL_SOURCE}")"
if [[ -z "${MODEL_DIR}" ]]; then
    exit 1
fi

if [[ ! -d "${MODEL_DIR}" ]]; then
    echo "Model directory does not exist: ${MODEL_DIR}" >&2
    exit 1
fi

if [[ -n "${USER_OUTPUT_DIR}" ]]; then
    OUTPUT_DIR="${USER_OUTPUT_DIR}"
else
    if [[ -n "${TEMP_MODEL_DIR}" ]]; then
        MODEL_ID_SAFE="$(normalize_repo_id "${MODEL_SOURCE}")"
        if [[ -z "${MODEL_ID_SAFE}" ]]; then
            MODEL_ID_SAFE="model"
        fi
        MODEL_ID_SAFE="${MODEL_ID_SAFE//\//__}"
        DEFAULT_GGUF_ROOT="${DEFAULT_GGUF_ROOT:-/workspace/gguf}"
        mkdir -p "${DEFAULT_GGUF_ROOT}" 2>/dev/null || true
        OUTPUT_DIR="${DEFAULT_GGUF_ROOT}/${MODEL_ID_SAFE}"
        if [[ -z "${UPLOAD_REPO}" && -n "${CACHED_SOURCE_REPO_ID}" ]]; then
            if [[ "${CACHED_SOURCE_REPO_ID}" == */* ]]; then
                local_owner="${CACHED_SOURCE_REPO_ID%%/*}"
                local_name="${CACHED_SOURCE_REPO_ID#*/}"
                if [[ "${local_owner}" != "${CACHED_SOURCE_REPO_ID}" && -n "${local_owner}" && -n "${local_name}" ]]; then
                    UPLOAD_REPO="${local_owner}/${local_name}-gguf"
                fi
            fi
        fi
    else
        OUTPUT_DIR="${MODEL_DIR}"
    fi
fi

mkdir -p "${OUTPUT_DIR}"

GGUF_TYPE="Q4_K_M"
PRECISION_OUTTYPE="f16"
MODEL_BASENAME="model"

CONVERT_SCRIPT="${LLAMA_CPP_DIR}/convert-hf-to-gguf.py"
QUANT_BIN="${LLAMA_CPP_DIR}/build/bin/llama-quantize"

if [[ ! -f "${CONVERT_SCRIPT}" ]]; then
    echo "convert-hf-to-gguf.py not found at ${CONVERT_SCRIPT}. Run scripts/install_llama_cpp.sh first." >&2
    exit 1
fi

if [[ ! -x "${QUANT_BIN}" ]]; then
    echo "llama-quantize binary not found at ${QUANT_BIN}. Build llama.cpp with scripts/install_llama_cpp.sh." >&2
    exit 1
fi

FLOAT_GGUF="${OUTPUT_DIR}/${MODEL_BASENAME}-${PRECISION_OUTTYPE}.gguf"
QUANT_GGUF="${OUTPUT_DIR}/${MODEL_BASENAME}-${GGUF_TYPE}.gguf"

if [[ ! -f "${FLOAT_GGUF}" ]]; then
    echo ">>> Converting Hugging Face model from ${MODEL_DIR} to GGUF (${PRECISION_OUTTYPE})"
    python3 "${CONVERT_SCRIPT}" \
        --model-dir "${MODEL_DIR}" \
        --outfile "${FLOAT_GGUF}" \
        --outtype "${PRECISION_OUTTYPE}"
else
    echo ">>> Reusing existing GGUF baseline at ${FLOAT_GGUF}"
fi

echo ">>> Quantizing ${FLOAT_GGUF} -> ${QUANT_GGUF} (Q4_K_M)"
"${QUANT_BIN}" "${FLOAT_GGUF}" "${QUANT_GGUF}" "${GGUF_TYPE}"

echo ">>> Quantized model stored at ${QUANT_GGUF}"

if [[ -n "${UPLOAD_REPO}" ]]; then
    ensure_hf_cli
    ensure_hf_login
    ensure_hf_repo "${UPLOAD_REPO}"

    remote_prefix="${UPLOAD_PREFIX%/}"
    if [[ -n "${remote_prefix}" ]]; then
        remote_quant_path="${remote_prefix}/$(basename "${QUANT_GGUF}")"
        remote_float_path="${remote_prefix}/$(basename "${FLOAT_GGUF}")"
    else
        remote_quant_path="$(basename "${QUANT_GGUF}")"
        remote_float_path="$(basename "${FLOAT_GGUF}")"
    fi

    upload_file_to_hf "${UPLOAD_REPO}" "${QUANT_GGUF}" "${remote_quant_path}"

    if [[ "${UPLOAD_INCLUDE_FLOAT}" == "1" ]]; then
        upload_file_to_hf "${UPLOAD_REPO}" "${FLOAT_GGUF}" "${remote_float_path}"
    fi

    echo ">>> Готово. Проверьте репозиторий: https://huggingface.co/${UPLOAD_REPO}"
else
    echo ">>> Переменная HF_UPLOAD_REPO не задана. Пропускаем выгрузку на Hugging Face."
fi
