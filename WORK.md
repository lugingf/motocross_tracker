# Work Notes

## Main Entry

Use the project environment:

```bash
./myenv/bin/python -m mx_tracker --help
```

## GoPro Preprocessing

GoPro clips must be ordered by clip id and chapter, for example:

```text
GX010556 -> GX020556 -> GX030556 -> GX010557
```

Build a concat list in GoPro order:

```bash
./myenv/bin/python -m mx_tracker gopro list \
  --pattern "$HOME/Movies/GoPro/StartTakeOutForDetection/GX*.MP4" \
  --output-list files.txt
```

Or:

```bash
make gopro-list \
  GOPRO_PATTERN="$HOME/Movies/GoPro/StartTakeOutForDetection/GX*.MP4" \
  CONCAT_LIST=files.txt
```

Build list, merge clips and transcode to detector-friendly H.264:

```bash
./myenv/bin/python -m mx_tracker gopro prepare \
  --pattern "$HOME/Movies/GoPro/StartTakeOutForDetection/GX*.MP4" \
  --output-dir artifacts/gopro \
  --name session_01
```

Or:

```bash
make gopro-prepare \
  GOPRO_PATTERN="$HOME/Movies/GoPro/StartTakeOutForDetection/GX*.MP4" \
  GOPRO_OUTPUT_DIR=artifacts/gopro \
  GOPRO_NAME=session_01
```

If clips are spread across nested folders:

```bash
./myenv/bin/python -m mx_tracker gopro prepare \
  --pattern "$HOME/Movies/GoPro/**/*.MP4" \
  --recursive \
  --output-dir artifacts/gopro \
  --name full_day
```

Fallback manual ffmpeg targets:

```bash
make ffmpeg-transcode RAW_SOURCE=~/Movies/GoPro/GX010473.MP4 TRANSCODE_OUTPUT=artifacts/GX010473_h264.mp4
make ffmpeg-concat-list CONCAT_GLOB="$HOME/Movies/GoPro/GX*.MP4" CONCAT_LIST=files.txt
make ffmpeg-concat CONCAT_LIST=files.txt CONCAT_OUTPUT=artifacts/merged.mp4
make ffmpeg-prepare CONCAT_OUTPUT=artifacts/merged.mp4 PREP_OUTPUT=artifacts/merged_h264.mp4
```

## Finish Line

Pick the finish line on the first frame:

```bash
./myenv/bin/python -m mx_tracker line calibrate --source race.mp4
```

## Data Collection

Collect bike crops from a file:

```bash
./myenv/bin/python -m mx_tracker collect \
  --config configs/default.yaml \
  --source artifacts/gopro/session_01_merged_h264.mp4 \
  --out-dir data/raw/session_01
```

Collect from a stream:

```bash
./myenv/bin/python -m mx_tracker collect \
  --config configs/default.yaml \
  --source rtsp://camera/stream \
  --source-mode stream \
  --out-dir data/raw/live_session
```

## Dataset Build

```bash
./myenv/bin/python -m mx_tracker dataset build \
  --raw-dir data/raw/session_01 \
  --dataset-dir data/datasets/session_01 \
  --clean
```

Validate:

```bash
./myenv/bin/python -m mx_tracker dataset validate \
  --dataset-dir data/datasets/session_01
```

## Training

```bash
./myenv/bin/python -m mx_tracker train \
  --data-yaml data/datasets/session_01/data.yaml \
  --model-path yolov8n.pt \
  --project-dir runs/detect \
  --run-name mx_session_01 \
  --epochs 120 \
  --imgsz 640
```

## Detection

File mode:

```bash
./myenv/bin/python -m mx_tracker detect file \
  --config configs/default.yaml \
  --source artifacts/gopro/session_01_merged_h264.mp4 \
  --out-dir artifacts/race_01
```

Stream mode:

```bash
./myenv/bin/python -m mx_tracker detect stream \
  --config configs/default.yaml \
  --source rtsp://camera/stream \
  --out-dir artifacts/live_01
```

Replay a local file as a live stream:

```bash
./myenv/bin/python -m mx_tracker detect stream \
  --config configs/default.yaml \
  --source artifacts/gopro/session_01_merged_h264.mp4 \
  --loop-source \
  --out-dir artifacts/replay_01
```

## Service

Start the local HTTP service:

```bash
./myenv/bin/python -m mx_tracker serve --config configs/default.yaml
```

## Video Baseline

Minimum practical input:

```text
1920x1080, 30 fps, plate width >= 50-60 px, plate height >= 20-24 px
```

Recommended:

```text
2560x1440 or 3840x2160, 50-60 fps, plate width >= 80-120 px
```
