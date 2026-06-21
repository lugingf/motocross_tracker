# MX Tracker

Инструкции по использованию проекта для:

- сбора новых данных;
- сборки датасета;
- обучения модели номерных табличек;
- детекции из файла;
- детекции из входящего потока;
- пост-обработки и пересчёта кругов;
- запуска HTTP-сервиса.

## Требования

Проект рассчитан на запуск в существующем окружении:

```bash
./myenv/bin/python -m mx_tracker --help
```

Базовые зависимости перечислены в [requirements.txt](requirements.txt).

## Структура запуска

Основной вход:

```bash
./myenv/bin/python -m mx_tracker ...
```

Совместимый запуск через старый файл:

```bash
./myenv/bin/python main.py ...
```

Доступные команды:

- `config` — инициализация конфига
- `gopro` — подготовка GoPro видео
- `line` — калибровка финишной линии
- `collect` — сбор кропов для обучения
- `dataset` — сборка и валидация датасета
- `train` — обучение модели
- `detect` — детекция пересечений из файла или потока
- `reid-watch` — фоновая дорезолюция unresolved пересечений
- `recount` — пересчёт кругов из events.jsonl в results.csv
- `serve` — HTTP-сервис

## Типичный гоночный день

Полный процесс от видео до итоговой таблицы:

```
1. detect      →  events.jsonl  (факты пересечений, lap пустой)
2. reid-watch  →  дополняет events.jsonl (пока detect работает или после)
3. ручная правка unresolved кропов (при необходимости)
4. recount     →  results.csv   (финальная таблица с lap и lap_time)
```

### Терминал 1 — детекция

```bash
./myenv/bin/python -m mx_tracker detect file \
  --config configs/default.yaml \
  --source race.mp4 \
  --out-dir artifacts/race_01
```

### Терминал 2 — фоновая дорезолюция (параллельно)

```bash
./myenv/bin/python -m mx_tracker reid-watch \
  --run-dir artifacts/race_01 \
  --plate-model data/models/yolov8n_plates_ft1.pt \
  --plate-conf-low 0.15
```

### После гонки — пересчёт кругов

```bash
./myenv/bin/python -m mx_tracker recount --run-dir artifacts/race_01
```

Результат: `artifacts/race_01/results.csv` с правильными `lap` и `lap_time`.

## Подготовка видео через ffmpeg

Для сырых файлов GoPro preprocessing часто нужен обязательно. Типичные причины:

- исходник закодирован в формате, который OpenCV/YOLO читает нестабильно;
- слишком тяжёлый HEVC/H.265 поток;
- несколько клипов нужно сначала склеить;
- нужен более удобный H.264 MP4 для детекции и обучения.

Если сырой файл нормально читается и стабильно идёт в пайплайне, этот шаг можно пропустить.

### Автоматическая подготовка GoPro в правильном порядке

Для клипов GoPro желательно использовать GoPro-aware сортировку, а не обычный `sort`.

Правильный порядок:

```text
GX010556, GX020556, GX030556, GX010557
```

Неправильный лексикографический порядок:

```text
GX010556, GX010557, GX020556, GX030556
```

Построить concat-list в GoPro порядке:

```bash
./myenv/bin/python -m mx_tracker gopro list \
  --pattern "$HOME/Movies/GoPro/StartTakeOutForDetection/GX*.MP4" \
  --output-list files.txt
```

Или через `make`:

```bash
make gopro-list \
  GOPRO_PATTERN="$HOME/Movies/GoPro/StartTakeOutForDetection/GX*.MP4" \
  CONCAT_LIST=files.txt
```

Сразу построить list, склеить и перекодировать:

```bash
./myenv/bin/python -m mx_tracker gopro prepare \
  --pattern "$HOME/Movies/GoPro/StartTakeOutForDetection/GX*.MP4" \
  --output-dir artifacts/gopro \
  --name session_01
```

Или через `make`:

```bash
make gopro-prepare \
  GOPRO_PATTERN="$HOME/Movies/GoPro/StartTakeOutForDetection/GX*.MP4" \
  GOPRO_OUTPUT_DIR=artifacts/gopro \
  GOPRO_NAME=session_01
```

Результат:

- `session_01_concat.txt`
- `session_01_merged.mp4`
- `session_01_merged_h264.mp4`

Если файлы лежат по подпапкам, добавь recursive glob и флаг `--recursive`:

```bash
./myenv/bin/python -m mx_tracker gopro prepare \
  --pattern "$HOME/Movies/GoPro/**/*.MP4" \
  --recursive \
  --output-dir artifacts/gopro \
  --name full_day
```

После этого в `collect` или `detect` обычно подаётся уже `*_merged_h264.mp4`.

### Низкоуровневые ffmpeg-команды

Если нужен полностью ручной контроль, остаются generic ffmpeg-таргеты.

### Перекодировать один файл вручную

```bash
make ffmpeg-transcode \
  RAW_SOURCE=~/Movies/GoPro/GX010473.MP4 \
  TRANSCODE_OUTPUT=artifacts/input_1_h264.mp4
```

Полезные переменные:

- `CRF=18`
- `PRESET=veryfast`
- `FPS=60`
- `SCALE=1920:1080`
- `VIDEO_CODEC=libx264`
- `AUDIO_CODEC=copy`

### Построить generic список файлов для concat

```bash
make ffmpeg-concat-list \
  CONCAT_GLOB="$HOME/Movies/GoPro/StartTakeOutForDetection/GX*.MP4" \
  CONCAT_LIST=files.txt
```

### Склеить клипы без перекодирования

```bash
make ffmpeg-concat \
  CONCAT_LIST=files.txt \
  CONCAT_OUTPUT=artifacts/merged.mp4
```

### Подготовить итоговый файл для детектора

Из одного файла:

```bash
make ffmpeg-prepare \
  RAW_SOURCE=~/Movies/GoPro/GX010473.MP4 \
  PREP_OUTPUT=artifacts/GX010473_h264.mp4
```

Из уже склеенного файла:

```bash
make ffmpeg-prepare \
  CONCAT_OUTPUT=artifacts/merged.mp4 \
  PREP_OUTPUT=artifacts/merged_h264.mp4
```

## Конфиг

Базовый конфиг лежит в [configs/default.yaml](configs/default.yaml).

Сгенерировать локальную копию:

```bash
./myenv/bin/python -m mx_tracker config init --output configs/local.yaml
```

Основные секции конфига:

- `runtime`: устройство и fallback FPS.
- `line`: финишная линия, ширина зоны, cooldown и направление (`left_to_right` или `right_to_left`).
- `models`: модель мотоциклов, модель `plate + digits`, трекер и пороги.
- `reads`: частота чтения номера и окно агрегации.
- `reid`: опциональный fallback через галерею.
- `stream`: поведение потока и переподключения.
- `output`: какие артефакты сохранять.
- `service`: host и port HTTP-сервиса.

Относительные пути можно задавать либо относительно файла конфига, либо относительно корня репозитория.

## Калибровка финишной линии

Выбрать линию на первом кадре:

```bash
./myenv/bin/python -m mx_tracker line calibrate --source race.mp4
```

Команда вернёт строку формата:

```text
12.34%,78.90%,88.76%,35.43%
```

Её можно вставить в `line.value` в YAML или передать через `--line`.

Направление задаётся в конфиге: `line.direction: left_to_right` или `right_to_left`.

## Пайплайн обучения на новых данных

### 1. Собрать кропы мотоциклов

Из файла:

```bash
./myenv/bin/python -m mx_tracker collect \
  --config configs/default.yaml \
  --source race.mp4 \
  --out-dir data/raw/session_01
```

Из входящего потока:

```bash
./myenv/bin/python -m mx_tracker collect \
  --config configs/default.yaml \
  --source rtsp://camera/stream \
  --source-mode stream \
  --out-dir data/raw/live_session
```

Полезные опции:

- `--limit-frames N`: ограничить прогон.
- `--calibrate-line`: выбрать линию на первом кадре.
- `--line ...`: временно переопределить линию.
- `--loop-source`: крутить локальный файл как поток.
- `--reconnect-delay`: задержка перед переподключением.
- `--max-reconnects`: максимум переподключений, `-1` для бесконечного режима.

Результат в `out-dir`:

- `frame*_tid*.jpg`: кропы мотоциклов;
- `events.csv`: таблица с координатами и путём к кропу;
- `events.jsonl`: те же события в JSONL;
- `summary.json`: итоговая сводка;
- `overlay.mp4`: видео с наложениями, если включено в конфиге.

### 2. Разметить изображения

Размечай `jpg` в YOLO-совместимом инструменте (рекомендуется makesense.ai). Для текущей схемы используются классы:

```text
plate, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
```

Правила:

- `txt` должен лежать рядом с `jpg`;
- имя `txt` должно совпадать с именем картинки;
- формат строки: `class cx cy w h`;
- координаты нормализованы в диапазоне `0..1`;
- `fliplr=0.0` обязателен при обучении — цифры нельзя зеркалить.

Если в кадре есть мотоцикл, но номер не виден, можно оставить изображение без `txt` и потом включить его как negative sample через `--include-unlabeled`.

### 3. Собрать train/val датасет

```bash
./myenv/bin/python -m mx_tracker dataset build \
  --raw-dir data/raw/session_01 \
  --dataset-dir data/datasets/session_01 \
  --clean
```

Полезные опции:

- `--train-ratio 0.8`
- `--seed 0`
- `--clean`: удалить старый dataset dir перед сборкой
- `--include-unlabeled`: включить изображения без разметки
- `--classes ...`: переопределить список классов

Проверить согласованность:

```bash
./myenv/bin/python -m mx_tracker dataset validate \
  --dataset-dir data/datasets/session_01
```

Проверка показывает:

- количество train/val изображений;
- количество label-файлов;
- missing labels;
- orphan labels;
- overlap между train и val;
- распределение классов.

### 4. Обучить модель

```bash
./myenv/bin/python -m mx_tracker train \
  --data-yaml data/datasets/session_01/data.yaml \
  --model-path yolov8n.pt \
  --project-dir runs/detect \
  --run-name mx_session_01 \
  --epochs 120 \
  --imgsz 640 \
  --batch 16 \
  --device auto
```

Полезные параметры:

- `--epochs`
- `--imgsz`
- `--batch`
- `--device`
- `--workers`

Артефакты обучения сохраняются в `project-dir/run-name`.

## Детекция

### Из файла

```bash
./myenv/bin/python -m mx_tracker detect file \
  --config configs/default.yaml \
  --source race.mp4 \
  --out-dir artifacts/race_01
```

Полезные опции:

- `--limit-frames N`
- `--calibrate-line`
- `--line ...`
- `--line-width 24`
- `--vehicle-model path/to/model.pt`
- `--plate-model path/to/best.pt`
- `--device auto|cpu|cuda:0|mps`

### Из входящего потока

Поддерживаются:

- `rtsp://...`
- `http://...`
- локальный файл как live replay
- индекс камеры, например `0`

Запуск:

```bash
./myenv/bin/python -m mx_tracker detect stream \
  --config configs/default.yaml \
  --source rtsp://camera/stream \
  --out-dir artifacts/live_01
```

Локальный файл как поток:

```bash
./myenv/bin/python -m mx_tracker detect stream \
  --config configs/default.yaml \
  --source race.mp4 \
  --loop-source \
  --out-dir artifacts/replay_01
```

Для stream-режима дополнительно полезны:

- `--loop-source`
- `--reconnect-delay`
- `--max-reconnects`

### Что пишется в events.jsonl и events.csv

Каждое пересечение — это факт: кто, когда, с какой уверенностью. Поля `lap` и `lap_time` намеренно пустые — они заполняются командой `recount` после того как все unresolved разрешены.

Поле `identity_source`:

| Значение    | Смысл                                                   |
|-------------|---------------------------------------------------------|
| `plate`     | Номер распознан моделью в реальном времени              |
| `reid`      | Идентифицирован через визуальное сходство (live reid)   |
| `unresolved`| Не удалось определить во время гонки                    |
| `plate_reread` | Распознан reid-watch с пониженным порогом            |
| `reid_post` | Идентифицирован reid-watch через галерею                |
| `manual`    | Номер указан вручную в JSON-сайдкаре                    |

Поле `crop_file` содержит путь к сохранённому JPG кропу (если `save_plate_crops: true` в конфиге).

## Дорезолюция unresolved пересечений (reid-watch)

`reid-watch` запускается параллельно с детекцией или после. Он опрашивает `plate_crops/unresolved/` и пытается разрешить каждый кроп в порядке приоритета:

1. **Ручная аннотация** — JSON-сайдкар с заполненным `manual_plate` и `save: 1`
2. **Plate re-detection** — модель с пониженным порогом уверенности
3. **ReID** — визуальное сходство с галереей из `plate_crops/resolved/`

```bash
./myenv/bin/python -m mx_tracker reid-watch \
  --run-dir artifacts/race_01 \
  --plate-model data/models/yolov8n_plates_ft1.pt \
  --plate-conf-low 0.15 \
  --threshold 0.60
```

Параметры:

- `--run-dir` — директория с артефактами детекции (обязательно)
- `--plate-model` — модель для повторного распознавания номера
- `--plate-conf-low` — порог уверенности для plate re-detection (по умолчанию `0.15`)
- `--threshold` — порог сходства для ReID (по умолчанию `0.60`)
- `--device` — устройство (`cpu`, `cuda:0`, `mps`)
- `--poll-interval` — интервал опроса в секундах (по умолчанию `2.0`)
- `--idle-timeout` — остановить после N секунд без новых матчей

Когда в галерее появляются новые resolved кропы, reid-watch автоматически перепроверяет все ранее неразрешённые пересечения с обновлённой галереей.

### Ручная правка unresolved

Для каждого нераспознанного пересечения сохраняется:

- `plate_crops/unresolved/frameXXXX_tidYYYY.jpg` — кроп мотоцикла
- `plate_crops/unresolved/frameXXXX_tidYYYY.json` — метаданные

Содержимое JSON:

```json
{
  "frame_index": 51624,
  "tracker_id": 6225,
  "plate_text": null,
  "last_read": { "text": "8", "confidence": 0.421 },
  "observations": [...],
  "manual_plate": "",
  "save": 0
}
```

Чтобы исправить: открой JSON, впиши номер и выстави флаг:

```json
  "manual_plate": "133",
  "save": 1
```

reid-watch подхватит это при следующем опросе и запишет событие с `identity_source: "manual"`.

## Пересчёт кругов (recount)

`events.jsonl` — это лог фактов. Лапы считаются отдельно, когда все unresolved разрешены:

```bash
./myenv/bin/python -m mx_tracker recount --run-dir artifacts/race_01
```

Результат: `artifacts/race_01/results.csv` — итоговая таблица, отсортированная по timestamp, с правильными `lap` и `lap_time` для каждого гонщика.

Команду можно запускать несколько раз — каждый раз она пересчитывает по актуальному `events.jsonl`. Это удобно, когда ручные правки добавляются постепенно.

Вывод в консоли:

```
[recount] 47 events → artifacts/race_01/results.csv
[recount] riders: 12, total crossings: 47
  plate_133: 5 lap(s)
  plate_27: 4 lap(s)
  ...
```

## Что делает детектор

Рабочая схема:

1. Детектор и трекер находят мотоцикл в кадре.
2. Из кропа мотоцикла модель находит `plate` и цифры.
3. Если `plate` найден, номер собирается только из цифр внутри таблички.
4. Чтения номера агрегируются по нескольким кадрам (vote window).
5. Пересечение финишной линии фиксируется только в заданном направлении.
6. Если в одном боксе трекера два набора цифр с большим разрывом — это два байка. Оба получают отдельное событие, второму прибавляется `+0.1 с`.

## HTTP-сервис

Запуск:

```bash
./myenv/bin/python -m mx_tracker serve --config configs/default.yaml
```

Эндпоинты:

- `GET /health`
- `GET /jobs`
- `GET /jobs/<job_id>`
- `POST /jobs`
- `POST /jobs/<job_id>/stop`

Поддерживаемые `action`:

- `detect_file`
- `detect_stream`
- `collect`
- `dataset_build`
- `dataset_validate`
- `train`

Пример job на детекцию файла:

```bash
curl -X POST http://127.0.0.1:8080/jobs \
  -H 'Content-Type: application/json' \
  -d '{
    "action": "detect_file",
    "config_path": "configs/default.yaml",
    "source": "race.mp4",
    "output_dir": "artifacts/service_race"
  }'
```

Пример job на поток:

```bash
curl -X POST http://127.0.0.1:8080/jobs \
  -H 'Content-Type: application/json' \
  -d '{
    "action": "detect_stream",
    "config_path": "configs/default.yaml",
    "source": "rtsp://camera/stream",
    "output_dir": "artifacts/service_live"
  }'
```

## Минимальные параметры видео

Для корректной детекции ориентируйся не только на разрешение кадра, а на фактический размер таблички в кадре.

Минимальный рабочий уровень:

- `1920x1080`
- `30 fps`
- ширина таблички в кадре не меньше `50-60 px`
- высота таблички не меньше `20-24 px`

Рекомендуемый уровень:

- `2560x1440` или `3840x2160`
- `50-60 fps`
- ширина таблички `80-120 px` и больше
- короткая выдержка без сильного motion blur

Практические правила:

- если номер в кропе мотоцикла читается глазами без зума, модель обычно тоже стабилизируется;
- если табличка занимает меньше примерно `0.5%` ширины исходного кадра, качество начнёт резко падать;
- на финишной линии и при быстром боковом движении `60 fps` заметно лучше `30 fps`.

## Быстрые команды через Makefile

```bash
make help
make gopro-list GOPRO_PATTERN="$HOME/Movies/GoPro/GX*.MP4" CONCAT_LIST=files.txt
make gopro-prepare GOPRO_PATTERN="$HOME/Movies/GoPro/GX*.MP4" GOPRO_OUTPUT_DIR=artifacts/gopro GOPRO_NAME=session_01
make ffmpeg-transcode RAW_SOURCE=~/Movies/GoPro/GX010473.MP4 TRANSCODE_OUTPUT=artifacts/GX010473_h264.mp4
make ffmpeg-concat-list CONCAT_GLOB="$HOME/Movies/GoPro/GX*.MP4" CONCAT_LIST=files.txt
make ffmpeg-concat CONCAT_LIST=files.txt CONCAT_OUTPUT=artifacts/merged.mp4
make ffmpeg-prepare CONCAT_OUTPUT=artifacts/merged.mp4 PREP_OUTPUT=artifacts/merged_h264.mp4
make config
make line SOURCE=race.mp4
make collect SOURCE=race.mp4 OUT_DIR=data/raw/session_01
make dataset RAW_DIR=data/raw/session_01 DATASET_DIR=data/datasets/session_01
make train DATA_YAML=data/datasets/session_01/data.yaml RUN_NAME=mx_session_01
make detect-file SOURCE=race.mp4 OUT_DIR=artifacts/race_01
make detect-stream SOURCE=rtsp://camera/stream OUT_DIR=artifacts/live_01
make serve
```
