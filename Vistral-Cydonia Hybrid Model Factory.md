# Vistral-Cydonia Hybrid Model Factory

Этот репозиторий предоставляет проверенный конвейер для создания гибридной модели Vistral ↔ Cydonia на базе `mergekit` и последующей выгрузки результата на Hugging Face.

## Состав конвейера

1. **Фаза 1: Гибридное слияние.**
   - `scripts/01_run_merge.sh` выполняет упрощённый SLERP-мерж между `Vikhrmodels/Vistral-24B-Instruct` и `TheDrummer/Cydonia-24B-v4.2.0`.
   - `scripts/run_dare_linear_profiled.sh` запускает профильный DARE-Linear мердж с использованием базовой модели `mistralai/Mistral-Small-3.2-24B-Instruct-2506`.
2. **Фаза 2: Выгрузка.**
   - `scripts/04_upload_merged.sh` собирает артефакты и загружает их в указанный репозиторий на Hugging Face.

Дополнительно предусмотрен скрипт `scripts/evaluate_prompts.py` для быстрой ручной проверки полученной модели.

## Требования

- Аккаунт [RunPod](https://runpod.io) или другой сервер с GPU (рекомендуется ≥48 GB VRAM).
- Аккаунт [Hugging Face](https://huggingface.co) с токеном на запись (`write`).
- Дисковое пространство: ~250 GB.

## Запуск

1. **Клонируйте репозиторий и установите зависимости.**
   ```bash
   git clone https://github.com/YourUsername/YourRepoName.git
   cd YourRepoName
   sudo ./setup.sh
   ```

2. **Настройте окружение для доступа к Hugging Face.**
   ```bash
   export HF_USERNAME="ваш-логин"
   export HF_TOKEN="hf_..."
   export HF_EMAIL="you@example.com"
   ```

3. **Запустите нужный вариант мержа.**
   ```bash
   ./scripts/01_run_merge.sh
   # либо
   ./scripts/run_dare_linear_profiled.sh
   ```

4. **(Опционально) Проверьте модель на примерах.**
   ```bash
   python3 scripts/evaluate_prompts.py --model /workspace/merged_model
   ```

5. **Выгрузите результат.**
   ```bash
   export MERGED_MODEL_REPO_NAME="Vistral-Cydonia-Merged"
   ./scripts/04_upload_merged.sh
   ```

6. **Готово!** Модель появится в вашем репозитории на Hugging Face.

7. **Не забудьте остановить под на RunPod.**
