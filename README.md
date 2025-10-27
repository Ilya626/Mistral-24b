# Mistral-24b Utilities

Этот репозиторий содержит набор утилит для слияния (merge) моделей семейства Mistral и быстрой выгрузки результатов на Hugging Face.

## Быстрый старт

### 1. Установка окружения
```bash
sudo ./setup.sh
```
Скрипт установит системные пакеты, Git LFS и минимальный набор Python-зависимостей (PyTorch, `transformers`, `mergekit`, `datasets`, `huggingface_hub`).

### 2. Авторизация в Hugging Face CLI (если требуется доступ к приватным моделям)
```bash
huggingface-cli login
```
Можно также заранее экспортировать токен окружением `export HF_TOKEN=...`.

### 3. Выполнение слияния моделей
В репозитории есть два готовых варианта мержа:

- `scripts/01_run_merge.sh` — упрощённый SLERP-мерж Vistral-24B и Cydonia-24B (результат в `/workspace/merged_model`).
- `scripts/run_dare_linear_profiled.sh` — продвинутый DARE-Linear профильный мердж с использованием базовой модели `Mistral-Small-3.2-24B-Instruct-2506` (результат в `/workspace/merged_dare_linear_profiled`).

Оба скрипта автоматически загружают исходные модели (требуется `HF_TOKEN`) и сохраняют результат в `/workspace/...`.

### 4. Быстрая качественная проверка (опционально)
Используйте `scripts/evaluate_prompts.py`, чтобы прогнать несколько промптов из датасета `grandmaster2` или из собственного файла:

```bash
python3 scripts/evaluate_prompts.py --model /workspace/merged_model
```

Скрипт сохранит ответы в каталоге `evaluation_logs`.

### 5. Выгрузка результата на Hugging Face
```bash
export HF_USERNAME=your-name
export HF_TOKEN=hf_xxx
export HF_EMAIL=you@example.com
export MERGED_MODEL_REPO_NAME=my-merged-model

./scripts/04_upload_merged.sh
```

Скрипт создаст (или обновит) репозиторий на Hugging Face и загрузит в него содержимое каталога с результатами мержа. Для альтернативных директорий (например, DARE-Linear) задайте `MERGED_MODEL_DIR` перед запуском.

## Структура
- `scripts/01_run_merge.sh` — SLERP-мерж Vistral-24B ↔ Cydonia-24B.
- `scripts/run_dare_linear_profiled.sh` — профильный DARE-Linear мердж для экспериментов.
- `scripts/04_upload_merged.sh` — загрузка итоговой модели на Hugging Face.
- `scripts/evaluate_prompts.py` — быстрая ручная проверка качества.
