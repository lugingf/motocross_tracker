PYTHON ?= ./myenv/bin/python
FFMPEG ?= ffmpeg
CONFIG ?= configs/default.yaml
SOURCE ?=
OUT_DIR ?=
RAW_SOURCE ?=
TRANSCODE_OUTPUT ?=
CONCAT_GLOB ?=
CONCAT_LIST ?= files.txt
CONCAT_OUTPUT ?=
PREP_OUTPUT ?=
GOPRO_PATTERN ?=
GOPRO_OUTPUT_DIR ?= data/artifacts/gopro
GOPRO_NAME ?= gopro
GOPRO_RECURSIVE ?=
CRF ?= 18
PRESET ?= veryfast
FPS ?=
SCALE ?=
VIDEO_CODEC ?= libx264
AUDIO_CODEC ?= copy
MODELS_TAG ?= models-v1
MODELS ?=
MODELS_NOTES ?=

.PHONY: help test gopro-list gopro-prepare ffmpeg-transcode ffmpeg-concat-list ffmpeg-concat ffmpeg-prepare config line collect dataset train detect-file detect-stream serve save-models get-models verify-models lock-models

test:
	$(PYTHON) -m pytest tests/ -v

help:
	@printf '%s\n' \
	"make gopro-list            - build ffmpeg concat list in GoPro clip order" \
	"make gopro-prepare         - build list, merge clips and transcode to H.264" \
	"make ffmpeg-transcode       - transcode raw GoPro video to stable H.264 MP4" \
	"make ffmpeg-concat-list     - build concat list from CONCAT_GLOB" \
	"make ffmpeg-concat          - merge clips from CONCAT_LIST into CONCAT_OUTPUT" \
	"make ffmpeg-prepare         - merge and/or transcode clips for detector input" \
	"make config                - write a local config template" \
	"make line SOURCE=video.mp4 - interactively pick the finish line" \
	"make collect SOURCE=...    - collect bike crops for labeling" \
	"make dataset               - build train/val dataset from RAW_DIR" \
	"make train                 - train a new plate model" \
	"make detect-file           - process a finite video file" \
	"make detect-stream         - process a live source or replay" \
	"make serve                 - start the HTTP service" \
	"make save-models           - upload data/models/*.pt to a GitHub Release + write models.lock" \
	"make get-models            - download the models named in models.lock" \
	"make verify-models         - check local models against models.lock (no network)" \
	"make lock-models           - rewrite models.lock from local files without uploading"

gopro-list:
	@test -n "$(GOPRO_PATTERN)" || (echo "GOPRO_PATTERN is required"; exit 1)
	$(PYTHON) -m mx_tracker gopro list --pattern "$(GOPRO_PATTERN)" --output-list "$(CONCAT_LIST)" $(if $(GOPRO_RECURSIVE),--recursive,)

gopro-prepare:
	@test -n "$(GOPRO_PATTERN)" || (echo "GOPRO_PATTERN is required"; exit 1)
	$(PYTHON) -m mx_tracker gopro prepare \
		--pattern "$(GOPRO_PATTERN)" \
		--output-dir "$(GOPRO_OUTPUT_DIR)" \
		--name "$(GOPRO_NAME)" \
		--ffmpeg "$(FFMPEG)" \
		--crf $(CRF) \
		--preset "$(PRESET)" \
		--video-codec "$(VIDEO_CODEC)" \
		--audio-codec "$(AUDIO_CODEC)" \
		$(if $(GOPRO_RECURSIVE),--recursive,) \
		$(if $(FPS),--fps $(FPS),) \
		$(if $(SCALE),--scale $(SCALE),)

ffmpeg-transcode:
	@test -n "$(RAW_SOURCE)" || (echo "RAW_SOURCE is required"; exit 1)
	@test -n "$(TRANSCODE_OUTPUT)" || (echo "TRANSCODE_OUTPUT is required"; exit 1)
	$(FFMPEG) -y -i "$(RAW_SOURCE)" \
		-c:v $(VIDEO_CODEC) \
		-crf $(CRF) \
		-preset $(PRESET) \
		$(if $(FPS),-r $(FPS),) \
		$(if $(SCALE),-vf scale=$(SCALE),) \
		-pix_fmt yuv420p \
		-movflags +faststart \
		-c:a $(AUDIO_CODEC) \
		"$(TRANSCODE_OUTPUT)"

ffmpeg-concat-list:
	@test -n "$(CONCAT_GLOB)" || (echo "CONCAT_GLOB is required"; exit 1)
	python3 -c 'from pathlib import Path; import glob; files=sorted(glob.glob("$(CONCAT_GLOB)")); Path("$(CONCAT_LIST)").write_text("".join(f"file '\''{Path(f).resolve()}'\''\\n" for f in files)); print(f"written {len(files)} entries to $(CONCAT_LIST)")'

ffmpeg-concat:
	@test -n "$(CONCAT_OUTPUT)" || (echo "CONCAT_OUTPUT is required"; exit 1)
	@test -f "$(CONCAT_LIST)" || (echo "CONCAT_LIST does not exist: $(CONCAT_LIST)"; exit 1)
	$(FFMPEG) -y -f concat -safe 0 -i "$(CONCAT_LIST)" -c copy "$(CONCAT_OUTPUT)"

ffmpeg-prepare:
	@test -n "$(PREP_OUTPUT)" || (echo "PREP_OUTPUT is required"; exit 1)
	@if [ -n "$(CONCAT_OUTPUT)" ]; then \
		$(FFMPEG) -y -i "$(CONCAT_OUTPUT)" \
			-c:v $(VIDEO_CODEC) \
			-crf $(CRF) \
			-preset $(PRESET) \
			$(if $(FPS),-r $(FPS),) \
			$(if $(SCALE),-vf scale=$(SCALE),) \
			-pix_fmt yuv420p \
			-movflags +faststart \
			-c:a $(AUDIO_CODEC) \
			"$(PREP_OUTPUT)"; \
	elif [ -n "$(RAW_SOURCE)" ]; then \
		$(FFMPEG) -y -i "$(RAW_SOURCE)" \
			-c:v $(VIDEO_CODEC) \
			-crf $(CRF) \
			-preset $(PRESET) \
			$(if $(FPS),-r $(FPS),) \
			$(if $(SCALE),-vf scale=$(SCALE),) \
			-pix_fmt yuv420p \
			-movflags +faststart \
			-c:a $(AUDIO_CODEC) \
			"$(PREP_OUTPUT)"; \
	else \
		echo "Set either RAW_SOURCE or CONCAT_OUTPUT"; \
		exit 1; \
	fi

config:
	$(PYTHON) -m mx_tracker config init --output configs/local.yaml

line:
	$(PYTHON) -m mx_tracker line calibrate --source $(SOURCE)

collect:
	$(PYTHON) -m mx_tracker collect --config $(CONFIG) $(if $(SOURCE),--source $(SOURCE),) $(if $(OUT_DIR),--out-dir $(OUT_DIR),)

dataset:
	$(PYTHON) -m mx_tracker dataset build --raw-dir $(RAW_DIR) --dataset-dir $(DATASET_DIR) --clean

train:
	$(PYTHON) -m mx_tracker train --data-yaml $(DATA_YAML) --run-name $(RUN_NAME)

detect-file:
	$(PYTHON) -m mx_tracker detect file --config $(CONFIG) $(if $(SOURCE),--source $(SOURCE),) $(if $(OUT_DIR),--out-dir $(OUT_DIR),)

detect-stream:
	$(PYTHON) -m mx_tracker detect stream --config $(CONFIG) $(if $(SOURCE),--source $(SOURCE),) $(if $(OUT_DIR),--out-dir $(OUT_DIR),)

serve:
	$(PYTHON) -m mx_tracker serve --config $(CONFIG)

save-models:
	$(PYTHON) utils/models.py save --tag $(MODELS_TAG) $(if $(MODELS_NOTES),--notes "$(MODELS_NOTES)",) $(MODELS)

get-models:
	$(PYTHON) utils/models.py get

verify-models:
	$(PYTHON) utils/models.py verify

lock-models:
	$(PYTHON) utils/models.py lock --tag $(MODELS_TAG) $(MODELS)
