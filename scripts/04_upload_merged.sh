#!/bin/bash
set -e

echo "--- ФАЗА 1Б: ВЫГРУЗКА СЛИТОЙ МОДЕЛИ НА HUGGING FACE ---"

MERGED_MODEL_DIR="${MERGED_MODEL_DIR:-/workspace/merged_model}"
HF_REPO_NAME="${MERGED_MODEL_REPO_NAME:-${NEW_MODEL_NAME}}"

if [ -z "$HF_TOKEN" ] || [ -z "$HF_USERNAME" ] || [ -z "$HF_EMAIL" ] || [ -z "$HF_REPO_NAME" ]; then
    cat <<USAGE
Ошибка: Требуются переменные окружения HF_TOKEN, HF_USERNAME, HF_EMAIL и MERGED_MODEL_REPO_NAME (или NEW_MODEL_NAME).
Пример:
  export HF_TOKEN=... # токен с правами write
  export HF_USERNAME=ваш_логин
  export HF_EMAIL=you@example.com
  export MERGED_MODEL_REPO_NAME=my-merged-model
USAGE
    exit 1
fi

if [ ! -d "$MERGED_MODEL_DIR" ]; then
    echo "Ошибка: Каталог с моделью $MERGED_MODEL_DIR не найден. Убедитесь, что фаза слияния выполнена."
    exit 1
fi

WORKSPACE_DIR="/workspace"
LOCAL_REPO="${WORKSPACE_DIR}/hf_repo_merged"

echo ">>> Создаём/обновляем репозиторий ${HF_USERNAME}/${HF_REPO_NAME}..."
huggingface-cli repo create "${HF_REPO_NAME}" --type model --exist-ok

rm -rf "${LOCAL_REPO}"
git clone "https://${HF_USERNAME}:${HF_TOKEN}@huggingface.co/${HF_USERNAME}/${HF_REPO_NAME}" "${LOCAL_REPO}"

cd "${LOCAL_REPO}"

echo ">>> Копируем артефакты модели из ${MERGED_MODEL_DIR}..."
rsync -av --delete --exclude ".git" "${MERGED_MODEL_DIR}/" "${LOCAL_REPO}/"

if [ -d ".git" ]; then
    git config user.name "${HF_USERNAME}"
    git config user.email "${HF_EMAIL}"
fi

echo ">>> Отслеживаем крупные файлы (*.safetensors)..."
git lfs track "*.safetensors" 2>/dev/null || true

git add .gitattributes || true
git add .

if git diff --cached --quiet; then
    echo ">>> Нет изменений для выгрузки."
else
    git commit -m "Upload merged model"
    echo ">>> Отправляем коммит на Hugging Face..."
    git push
fi

echo "--- ВЫГРУЗКА СЛИТОЙ МОДЕЛИ ЗАВЕРШЕНА ---"

echo ">>> Репозиторий доступен по ссылке: https://huggingface.co/${HF_USERNAME}/${HF_REPO_NAME}"
