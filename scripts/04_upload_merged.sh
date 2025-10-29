#!/bin/bash
set -e

echo "--- Р¤РђР—Рђ 1Р‘: Р’Р«Р“Р РЈР—РљРђ РЎР›РРўРћР™ РњРћР”Р•Р›Р РќРђ HUGGING FACE ---"

MERGED_MODEL_DIR="${MERGED_MODEL_DIR:-/workspace/merged_model}"
HF_REPO_NAME="${MERGED_MODEL_REPO_NAME:-${NEW_MODEL_NAME}}"

if [ -z "$HF_TOKEN" ] || [ -z "$HF_USERNAME" ] || [ -z "$HF_EMAIL" ] || [ -z "$HF_REPO_NAME" ]; then
    cat <<USAGE
РћС€РёР±РєР°: РўСЂРµР±СѓСЋС‚СЃСЏ РїРµСЂРµРјРµРЅРЅС‹Рµ РѕРєСЂСѓР¶РµРЅРёСЏ HF_TOKEN, HF_USERNAME, HF_EMAIL Рё MERGED_MODEL_REPO_NAME (РёР»Рё NEW_MODEL_NAME).
РџСЂРёРјРµСЂ:
  export HF_TOKEN=... # С‚РѕРєРµРЅ СЃ РїСЂР°РІР°РјРё write
  export HF_USERNAME=РІР°С€_Р»РѕРіРёРЅ
  export HF_EMAIL=you@example.com
  export MERGED_MODEL_REPO_NAME=my-merged-model
USAGE
    exit 1
fi

if [ ! -d "$MERGED_MODEL_DIR" ]; then
    echo "РћС€РёР±РєР°: РљР°С‚Р°Р»РѕРі СЃ РјРѕРґРµР»СЊСЋ $MERGED_MODEL_DIR РЅРµ РЅР°Р№РґРµРЅ. РЈР±РµРґРёС‚РµСЃСЊ, С‡С‚Рѕ С„Р°Р·Р° СЃР»РёСЏРЅРёСЏ РІС‹РїРѕР»РЅРµРЅР°."
    exit 1
fi

WORKSPACE_DIR="/workspace"
LOCAL_REPO="${WORKSPACE_DIR}/hf_repo_merged"

echo ">>> РЎРѕР·РґР°С‘Рј/РѕР±РЅРѕРІР»СЏРµРј СЂРµРїРѕР·РёС‚РѕСЂРёР№ ${HF_USERNAME}/${HF_REPO_NAME}..."
huggingface-cli repo create "${HF_REPO_NAME}" --type model --exist-ok

rm -rf "${LOCAL_REPO}"
git clone "https://${HF_USERNAME}:${HF_TOKEN}@huggingface.co/${HF_USERNAME}/${HF_REPO_NAME}" "${LOCAL_REPO}"

cd "${LOCAL_REPO}"

echo ">>> РљРѕРїРёСЂСѓРµРј Р°СЂС‚РµС„Р°РєС‚С‹ РјРѕРґРµР»Рё РёР· ${MERGED_MODEL_DIR}..."
rsync -av --delete --exclude ".git" "${MERGED_MODEL_DIR}/" "${LOCAL_REPO}/"

if [ -d ".git" ]; then
    git config user.name "${HF_USERNAME}"
    git config user.email "${HF_EMAIL}"
fi

echo ">>> РћС‚СЃР»РµР¶РёРІР°РµРј РєСЂСѓРїРЅС‹Рµ С„Р°Р№Р»С‹ (*.safetensors)..."
git lfs track "*.safetensors" 2>/dev/null || true
git lfs track "tokenizer.json" 2>/dev/null || true
git lfs track "tekken.json" 2>/dev/null || true

git add .gitattributes || true
git add .

if git diff --cached --quiet; then
    echo ">>> РќРµС‚ РёР·РјРµРЅРµРЅРёР№ РґР»СЏ РІС‹РіСЂСѓР·РєРё."
else
    git commit -m "Upload merged model"
    echo ">>> РћС‚РїСЂР°РІР»СЏРµРј РєРѕРјРјРёС‚ РЅР° Hugging Face..."
    git push
fi

echo "--- Р’Р«Р“Р РЈР—РљРђ РЎР›РРўРћР™ РњРћР”Р•Р›Р Р—РђР’Р•Р РЁР•РќРђ ---"

echo ">>> Р РµРїРѕР·РёС‚РѕСЂРёР№ РґРѕСЃС‚СѓРїРµРЅ РїРѕ СЃСЃС‹Р»РєРµ: https://huggingface.co/${HF_USERNAME}/${HF_REPO_NAME}"

