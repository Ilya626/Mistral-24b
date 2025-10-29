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

`01_run_merge.sh` принимает флаг `--profile` (или переменную окружения `MERGE_CONFIG`), чтобы использовать произвольный YAML-профиль — например, сгенерированный AIM-профайлером.

### 3.1. Построение AIM-профиля SLERP (опционально)
Чтобы получить детальный 40-слойный профиль на основе активаций обеих моделей, используйте `scripts/aim_profile.py`. Скрипт жёстко работает с датасетом `grandmaster2`, применяет единый Llama 3 шаблон чата и собирает статистику по «уверенным»/«неуверенным» токенам.

```bash
python3 scripts/aim_profile.py \
  --vistral-model /workspace/Vistral-24B-Instruct \
  --cydonia-model /workspace/Cydonia-24B-v4.2.0 \
  --output-dir profiles/grandmaster2
```

При необходимости можно задать `--base-model`, если SLERP должен стартовать не от Vistral, а от другого чекпоинта.

Для быстрой проверки работоспособности скрипта можно ограничиться одной моделью:

```bash
python3 scripts/aim_profile.py --vistral-model /workspace/Vistral-24B-Instruct --single-model vistral
```

В этом случае будет выгружена только статистика (`metrics.json`/`metrics.csv`), а профиль SLERP строиться не будет.

Чтобы не ждать прогон всего датасета, можно случайным образом выбрать небольшое подмножество промптов. Например, 20 штук:

```bash
python3 scripts/aim_profile.py \
  --vistral-model /workspace/Vistral-24B-Instruct \
  --single-model vistral \
  --sample-prompts 20 \
  --sample-seed 42
```

Флаг `--sample-prompts` вырезает указанное количество случайных промптов (после применения `--max-prompts` и `--dataset-split`), а `--sample-seed` позволяет детерминировать выборку.

На выходе формируется папка с отметкой времени, где лежат:

- `aim_profile.yml` — готовый профиль для `mergekit-yaml` (можно отредактировать вручную перед использованием);
- `metrics.json` — агрегированные метрики по всем 40 слоям (уверенность, неуверенность, перплексия);
- `metrics.csv` — табличное представление средних норм и количества токенов для быстрого анализа.

Затем можно запустить `01_run_merge.sh --profile profiles/grandmaster2/<timestamp>/aim_profile.yml` для слияния с учётом собранных весов.

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
- `scripts/aim_profile.py` — построение AIM-профиля для градиентного SLERP.
- `scripts/run_dare_linear_profiled.sh` — профильный DARE-Linear мердж для экспериментов.
- `scripts/04_upload_merged.sh` — загрузка итоговой модели на Hugging Face.
- `scripts/evaluate_prompts.py` — быстрая ручная проверка качества.

## Журнал проблем мержа и решения

| Проблема | Как проявлялась | Как решили |
| --- | --- | --- |
| Сценарии перезаписывали объединённый словарь `tokenizer.json` устаревшими `tokenizer.model`. | После мержа при квантовании или перезагрузке модель теряла токены из Cydonia. | Убрали флаг `--copy-tokenizer` и оставили mergekit собирать `tokenizer.json` с union-словарём Vistral + Cydonia. |
| Потеря метаданных токенайзера при мерже. | В итоговой папке отсутствовали `tokenizer_config.json`, `special_tokens_map.json` и другие sidecar-файлы, что ломало инструменты инференса. | Добавили функцию `copy_tokenizer_sidecars`, которая синхронизирует нужные JSON-файлы из исходных моделей только если их нет в целевой папке. |
| Невалидный объединённый словарь перед квантованием. | Квантовщики падали, потому что в `tokenizer.json` не было части токенов или специальных токенов. | Реализовали `validate_union_tokenizer` на чистом Python: он гарантирует присутствие всех токенов и спец-символов Vistral, Cydonia и базовой модели (для DARE). |
| Базовая модель DARE шла без токенайзера. | Запуск DARE-профиля останавливался, если каталогу `BASE_MODEL_DIR` не хватало файлов токенайзера. | Автоматически догружаем официальный токенайзер из `mistralai/Mistral-Small-3.1-24B-Base-2503`, если в базе нет `tokenizer.*`. |
| Некорректные `layer_range` в кастомных профилях. | Mergekit падал уже после выгрузки моделей из-за неверных диапазонов слоёв. | Добавили Python-проверку `verify_layer_ranges`, которая убеждается, что все `layer_range` покрывают ровно 40 слоёв и не имеют неверного шага. |
