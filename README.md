# Mistral-24b Utilities

Этот репозиторий содержит вспомогательные скрипты для подготовки и запуска больших языковых моделей.

## Быстрый старт

### 1. Установка зависимостей llama.cpp
```bash
./scripts/install_llama_cpp.sh
```
Скрипт скачает исходники `llama.cpp`, соберёт бинарники и установит Python-зависимости.

### 2. Авторизация в Hugging Face CLI (если требуется доступ к приватным моделям)
```bash
huggingface-cli login
```
Можно также заранее экспортировать токен окружением `export HF_TOKEN=...`.

### 3. Конвертация и квантование модели Hugging Face в GGUF (Q4_K_M)
Квантовать можно как локальную распакованную модель, так и указав ссылку/ID из Hugging Face.

```bash
# Пример для локальной папки
./scripts/run_quantize.sh /path/to/local/hf-model [/path/to/output]

# Пример для публичного репозитория
./scripts/run_quantize.sh TheBloke/Mistral-7B-Instruct-v0.2 ./quantized/mistral-7b

# Пример для URL на huggingface.co
./scripts/run_quantize.sh https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2
```
- `MODEL_SOURCE` — путь, repo ID или URL Hugging Face.
- `OUTPUT_DIR` (опционально) — папка для GGUF. Для скачанных моделей по умолчанию используется `artifacts/<repo_id>` внутри репозитория, для локальных — сама папка модели.

Скрипт автоматически скачает модель (если передан ID/URL), создаст float16-базу и квантованный файл в формате `Q4_K_M`.

Чтобы автоматически выгрузить результаты на Hugging Face, укажите репозиторий через переменную окружения и (опционально) настро
йте директорию загрузки:

```bash
export HF_UPLOAD_REPO="your-username/your-model-gguf"   # репозиторий, в который нужно заливать артефакты
export HF_UPLOAD_PREFIX="gguf"                           # (опционально) подпапка внутри репозитория
export HF_UPLOAD_INCLUDE_FLOAT=0                         # (опционально) 1 — грузить также float16 базовый GGUF

./scripts/run_quantize.sh TheBloke/Mistral-7B-Instruct-v0.2
```

Скрипт проверит авторизацию, при необходимости создаст репозиторий (по умолчанию приватный, флаг `HF_UPLOAD_PRIVATE=0` делает е
го публичным) и зальёт квантованные веса в указанный путь.

## Структура
- `scripts/install_llama_cpp.sh` — установка `llama.cpp` и зависимостей.
- `scripts/run_quantize.sh` — конвертация Hugging Face модели в GGUF и квантование в `Q4_K_M` с поддержкой скачивания по ссылке/ID и выгрузки результата на Hugging Face.
