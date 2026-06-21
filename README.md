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
2. reid-watch  →  дополняет events.jsonl + автоматически обновляет results.csv
3. ручная правка unresolved кропов (при необходимости)
                  reid-watch подхватывает правку и снова обновляет results.csv
```

`results.csv` актуален всегда, пока работает reid-watch. Запускать `recount` вручную не нужно.

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

reid-watch опрашивает `events.jsonl` каждый цикл — если файл изменился (новые пересечения от детектора или новые resolved события), автоматически вызывает `recount` и обновляет `results.csv`.

### Пересчёт вручную (при необходимости)

Если reid-watch не запускался или нужно пересчитать отдельно:

```bash
./myenv/bin/python -m mx_tracker recount --run-dir artifacts/race_01
# или напрямую:
./myenv/bin/python utils/recount.py --run-dir artifacts/race_01
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

Базовый конфиг детекции лежит в [configs/default.yaml](configs/default.yaml).  
Для каждого скрипта есть отдельный конфиг: `configs/reid_watch.yaml`, `configs/recount.yaml`, `configs/train.yaml`.

Сгенерировать пустую копию:

```bash
./myenv/bin/python -m mx_tracker config init --output configs/local.yaml
```

Относительные пути разрешаются относительно корня репозитория (где лежит конфиг).

---

### configs/default.yaml — детекция

#### Быстрый старт: source и out_dir

Чтобы не вводить `--source` и `--out-dir` каждый раз, раскомментируй в начале файла:

```yaml
source: data/videos/race.mp4
out_dir: data/artifacts/race_name
```

CLI-аргументы всегда имеют приоритет над значениями в конфиге.

---

#### `runtime`

| Параметр              | По умолчанию | Описание                                                                                 |
|-----------------------|--------------|------------------------------------------------------------------------------------------|
| `device`              | `auto`       | Устройство для YOLO: `auto` (выбирает лучшее), `cpu`, `cuda:0`, `mps` (Apple Silicon).   |
| `source_fps_fallback` | `30.0`       | FPS, используемый если OpenCV не может прочитать его из файла. Для обычных MP4 не нужен. |

---

#### `line` — финишная линия

| Параметр                   | По умолчанию       | Описание                                                                                                                                              |
|----------------------------|--------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| `value`                    | `"50%,5%,50%,95%"` | Координаты линии в пикселях (`x1,y1,x2,y2`) или процентах (`50%,5%,50%,95%`). Получить командой `line calibrate`.                                     |
| `width`                    | `24`               | Ширина зоны пересечения в пикселях. Байк засчитывается только если его центр попадает в эту зону.                                                     |
| `cooldown_sec`             | `2.0`              | Минимальная пауза между двумя пересечениями одного трека. Защита от двойного счёта.                                                                   |
| `direction`                | `left_to_right`    | Направление движения, которое засчитывается: `left_to_right` или `right_to_left`. Определяется по камере — байк едет слева направо или справа налево. |
| `read_distance_multiplier` | `2.5`              | Ширина зоны, в которой включается чтение номера: `width × multiplier`. При значении `2.5` и `width=24` — зона 60 px в каждую сторону от линии.        |

---

#### `models` — модели и детекция

| Параметр              | По умолчанию                        | Описание                                                                                                                                                                                                                  |
|-----------------------|-------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `vehicle_model`       | `data/models/yolov8n.pt`            | YOLO-модель для детекции и трекинга мотоциклов.                                                                                                                                                                           |
| `plate_model`         | `data/models/yolov8n_plates_ft1.pt` | YOLO-модель для распознавания номерных табличек и цифр.                                                                                                                                                                   |
| `tracker`             | `botsort.yaml`                      | Конфиг трекера для YOLO. По умолчанию встроенный BotSort. Для тонкой настройки можно указать свой файл, например `configs/botsort.yaml`.                                                                                  |
| `motorcycle_class_id` | `3`                                 | ID класса мотоцикла в COCO (3 = motorcycle).                                                                                                                                                                              |
| `vehicle_conf`        | `0.35`                              | Минимальная уверенность детектора для регистрации мотоцикла. Ниже — больше ложных срабатываний, выше — риск пропустить байк.                                                                                              |
| `vehicle_iou`         | `0.5`                               | Порог IoU для NMS при детекции мотоциклов.                                                                                                                                                                                |
| `vehicle_imgsz`       | `640`                               | Размер входного изображения для YOLO при детекции мотоциклов. Типичные значения: `320`, `480`, `640`, `960`, `1280`. Чем больше — тем выше точность на мелких объектах, но медленнее. На MPS хорошо работает `640`–`960`. |
| `plate_conf`          | `0.15`                              | Минимальная уверенность для распознавания цифр на табличке. Намеренно низкое — цифры считываются при любом угле и расстоянии.                                                                                             |
| `plate_has_class`     | `true`                              | Если `true`, модель выдаёт отдельный класс 0 для рамки таблички. Цифры — классы 1–10 (digit = class_id − 1). Если `false`, все классы 0–9 — сразу цифры.                                                                  |
| `plate_class_id`      | `0`                                 | ID класса рамки таблички в plate-модели (работает только при `plate_has_class: true`).                                                                                                                                    |
| `bike_crop_expand`    | `0.10`                              | Расширение кропа мотоцикла перед подачей в plate-модель: `1.0 + expand`. При `0.10` — увеличение на 10%.                                                                                                                  |
| `plate_box_expand`    | `0.12`                              | Расширение рамки таблички при вырезании.                                                                                                                                                                                  |
| `min_bike_crop_px`    | `96`                                | Минимальная сторона кропа байка в пикселях. Кропы меньше этого размера пропускаются — слишком мало для распознавания номера.                                                                                              |
| `plate_zone_n`        | `9`                                 | Количество зон для разбивки кропа байка перед plate-моделью: `1`, `2`, `4`, `9`, `32`. При `9` кроп делится на сетку 3×3.                                                                                                 |
| `plate_zone_select`   | `[4, 5, 7, 8]`                      | Какие зоны из `plate_zone_n` подавать в plate-модель (нумерация слева направо, сверху вниз, с 1). При `9` и `[4,5,7,8]` берётся нижняя половина кропа — там обычно находится номер.                                       |

---

#### `reads` — агрегация чтений номера

| Параметр              | По умолчанию | Описание                                                                                               |
|-----------------------|--------------|--------------------------------------------------------------------------------------------------------|
| `scan_every_n_frames` | `1`          | Читать номер раз в N кадров (только когда байк в зоне линии). `1` = каждый кадр.                       |
| `min_digits`          | `1`          | Минимальное количество цифр для признания чтения валидным.                                             |
| `vote_window_sec`     | `2.5`        | Окно голосования: при пересечении берётся самый частый номер среди чтений за последние N секунд.       |
| `track_state_ttl_sec` | `10.0`       | Время жизни состояния трека. Если байк не виден N секунд — его история (номера, позиция) сбрасывается. |

---

#### `reid` — визуальная идентификация в реальном времени

> ⚠️ Рекомендуется держать `enabled: false`. В текущей установке torchreid работает в режиме HSV-гистограмм, что даёт ненадёжные совпадения между байками в похожей экипировке. Нераспознанные пересечения лучше обрабатывать через `reid-watch` с plate-моделью.

| Параметр       | По умолчанию   | Описание                                                                               |
|----------------|----------------|----------------------------------------------------------------------------------------|
| `enabled`      | `false`        | Включить live-ReID при детекции.                                                       |
| `gallery_path` | `data/gallery` | Папка с поддиректориями по `rider_id`, каждая содержит JPG-кропы байка.                |
| `threshold`    | `0.6`          | Порог сходства для ReID-матча (0–1). В HSV-режиме автоматически поднимается до `0.70`. |

---

#### `stream` — поведение потока

| Параметр              | По умолчанию | Описание                                                           |
|-----------------------|--------------|--------------------------------------------------------------------|
| `reconnect_delay_sec` | `3.0`        | Пауза перед переподключением при разрыве (только `detect stream`). |
| `max_reconnects`      | `-1`         | Максимум переподключений. `-1` = бесконечно.                       |
| `loop_file`           | `false`      | Крутить локальный файл по кругу в stream-режиме.                   |
| `status_interval_sec` | `10.0`       | Частота вывода статус-строки в консоль (`frame=... events=...`).   |

---

#### `output` — артефакты

| Параметр           | По умолчанию | Описание                                                          |
|--------------------|--------------|-------------------------------------------------------------------|
| `write_video`      | `true`       | Записывать видео с оверлеем (`overlay.mp4`).                      |
| `write_csv`        | `true`       | Записывать `events.csv`.                                          |
| `write_jsonl`      | `true`       | Записывать `events.jsonl`.                                        |
| `write_summary`    | `true`       | Записывать `summary.json` по итогам.                              |
| `overlay_top_n`    | `10`         | Сколько rider_id показывать в левом верхнем углу видео.           |
| `save_plate_crops` | `true`       | Сохранять кропы таблички при каждом пересечении в `plate_crops/`. |

---

#### `service`

| Параметр | По умолчанию | Описание                         |
|----------|--------------|----------------------------------|
| `host`   | `127.0.0.1`  | Хост для HTTP-сервиса (`serve`). |
| `port`   | `8080`       | Порт для HTTP-сервиса.           |

---

### configs/reid_watch.yaml

Конфиг для `reid-watch`. Автоматически подхватывается без `--config`.

| Параметр         | Описание                                                                                     |
|------------------|----------------------------------------------------------------------------------------------|
| `run_dir`        | Директория с артефактами детекции (обязательно).                                             |
| `plate_model`    | Путь к plate-модели для повторного распознавания.                                            |
| `plate_conf_low` | Порог уверенности для plate-модели в reid-watch (обычно ниже чем при детекции, `0.10–0.20`). |
| `threshold`      | Порог сходства для ReID-матча.                                                               |
| `device`         | `cpu`, `cuda:0`, `mps`.                                                                      |
| `poll_interval`  | Интервал опроса директории в секундах.                                                       |
| `idle_timeout`   | Остановить после N секунд без новых матчей. Если не задан — работает до Ctrl-C.              |

---

### configs/recount.yaml

| Параметр  | Описание                     |
|-----------|------------------------------|
| `run_dir` | Директория с `events.jsonl`. |

---

### configs/train.yaml

| Параметр      | По умолчанию             | Описание                                        |
|---------------|--------------------------|-------------------------------------------------|
| `data_yaml`   | —                        | Путь к `data.yaml` датасета (обязательно).      |
| `model_path`  | `data/models/yolov8n.pt` | Базовый чекпоинт для дообучения.                |
| `project_dir` | `data/runs/detect`       | Куда сохраняются артефакты обучения.            |
| `run_name`    | `mx_plate_train`         | Имя прогона внутри `project_dir`.               |
| `epochs`      | `100`                    | Количество эпох.                                |
| `imgsz`       | `640`                    | Размер входного изображения при обучении.       |
| `batch`       | `16`                     | Размер батча. `-1` для автоматического подбора. |
| `device`      | `auto`                   | Устройство обучения.                            |
| `workers`     | `8`                      | Количество воркеров DataLoader.                 |

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

Каждое пересечение — это факт: кто, когда, с какой уверенностью. Поля `lap` и `lap_time` намеренно пустые — они заполняются `recount`.

Основные поля:

| Поле             | Описание                                                                 |
|------------------|--------------------------------------------------------------------------|
| `timestamp`      | Секунды от начала видео/стрима.                                           |
| `wall_time`      | Реальное время события в ISO-формате (`2026-06-21T10:30:15+00:00`). Вычисляется из `run_info.json` в папке прогона. Пусто для старых прогонов без этого файла. |
| `frame_index`    | Номер кадра.                                                             |
| `tracker_id`     | ID трека (YOLO BotSort). Может меняться между прогонами.                 |
| `rider_id`       | Устойчивый ID вида `plate_133` или пустой для unresolved.                |
| `identity_source`| Способ идентификации (см. таблицу ниже).                                 |
| `plate_text`     | Распознанный номер (строка цифр).                                        |
| `plate_conf`     | Уверенность распознавания.                                               |
| `lap`            | Номер круга. Заполняется recount.                                        |
| `lap_time`       | Время круга в секундах. Заполняется recount.                             |
| `center_x/y`     | Координаты центра байка в пикселях.                                      |
| `crop_file`      | Путь к JPG кропу (если `save_plate_crops: true`). При одновременном пересечении нескольких байков каждый получает кроп в своей папке. |

> В `events.csv` bbox-колонок нет — только `center_x/y`. Полные данные с `bbox` доступны в `events.jsonl`.

Поле `identity_source`:

| Значение       | Смысл                                                      |
|----------------|------------------------------------------------------------|
| `plate`        | Номер распознан моделью в реальном времени                 |
| `reid`         | Идентифицирован через визуальное сходство (live reid)      |
| `unresolved`   | Не удалось определить во время гонки                       |
| `plate_reread` | Распознан reid-watch с пониженным порогом                  |
| `reid_post`    | Идентифицирован reid-watch через галерею                   |
| `manual`       | Номер указан вручную в JSON-сайдкаре                       |

В папке прогона также создаётся `run_info.json` — содержит `started_at` (ISO, UTC) и `source`. Используется для вычисления `wall_time` в `recount` и `reid-watch`.

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

После каждого цикла, где `events.jsonl` изменился (новые пересечения от детектора или новые resolved события), reid-watch автоматически вызывает `recount` и обновляет `results.csv`.

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

`events.jsonl` — это лог фактов. `results.csv` — итоговая таблица с `lap` и `lap_time`. Обычно её обновляет reid-watch автоматически. Для ручного пересчёта:

```bash
./myenv/bin/python -m mx_tracker recount --run-dir artifacts/race_01
# или как standalone-скрипт:
./myenv/bin/python utils/recount.py --run-dir artifacts/race_01
```

`results.csv` не содержит bbox-колонок — только `timestamp`, `wall_time`, основные поля события, `center_x/y` и `crop_file`. Если в папке прогона есть `run_info.json`, каждой строке добавляется `wall_time`.

Команду можно запускать несколько раз — каждый раз она пересчитывает по актуальному `events.jsonl`.

#### lap_time первого круга

По умолчанию `race_start_sec=0` — lap 1 считается от начала видео. Если гонка стартовала позже:

```bash
# По смещению от начала видео (секунды)
python utils/recount.py --run-dir artifacts/race_01 --race-start-sec 47.5

# По времени суток (требует run_info.json в папке прогона)
python utils/recount.py --run-dir artifacts/race_01 --race-start-at "10:31:00"
```

Или зафиксировать в конфиге:

```yaml
# configs/recount.yaml
run_dir: data/artifacts/race_01
race_start_sec: 47.5
# race_start_at: "10:31:00"
```

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
