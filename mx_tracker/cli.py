from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import yaml

from .config import load_settings, write_default_config
from .geometry import pick_line_on_frame, to_percent_str
from .gopro import build_concat_list, prepare_gopro_video
from .pipeline import collect_samples, run_file_detection, run_stream_detection
from .recount import recount
from .reid_watch import run_reid_watch
from .service import run_service
from .training import build_dataset, train_model, validate_dataset


def _print_json(payload: dict[str, object]) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _load_script_config(config_path: str | None, *, required: bool = False) -> dict:
    """Load a per-script YAML config file, returning {} when not provided or not found."""
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Config file not found: {config_path}")
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _load_and_override_settings(args: argparse.Namespace):
    settings, base_dir = load_settings(getattr(args, "config", None))
    if getattr(args, "line", None):
        settings.line.value = args.line
    if getattr(args, "line_width", None) is not None:
        settings.line.width = args.line_width
    if getattr(args, "vehicle_model", None):
        settings.models.vehicle_model = args.vehicle_model
    if getattr(args, "plate_model", None):
        settings.models.plate_model = args.plate_model
    if getattr(args, "device", None):
        settings.runtime.device = args.device
    if getattr(args, "enable_reid", False):
        settings.reid.enabled = True
    if getattr(args, "disable_reid", False):
        settings.reid.enabled = False
    if getattr(args, "gallery", None):
        settings.reid.gallery_path = args.gallery
    if getattr(args, "loop_source", False):
        settings.stream.loop_file = True
    if getattr(args, "reconnect_delay", None) is not None:
        settings.stream.reconnect_delay_sec = args.reconnect_delay
    if getattr(args, "max_reconnects", None) is not None:
        settings.stream.max_reconnects = args.max_reconnects
    if getattr(args, "digits_only", False):
        settings.models.plate_has_class = False
    return settings, base_dir


def _handle_config_init(args: argparse.Namespace) -> int:
    path = write_default_config(args.output)
    print(path)
    return 0


def _handle_line_calibrate(args: argparse.Namespace) -> int:
    capture = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open source: {args.source}")
    try:
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError("Cannot read first frame")
        picked = pick_line_on_frame(frame)
        if picked is None:
            raise RuntimeError("Line calibration cancelled")
        height, width = frame.shape[:2]
        (x1, y1), (x2, y2) = picked
        print(to_percent_str(x1, y1, x2, y2, width, height))
    finally:
        capture.release()
    return 0


def _handle_gopro_list(args: argparse.Namespace) -> int:
    result = build_concat_list(
        pattern=args.pattern,
        output_list=args.output_list,
        recursive=args.recursive,
    ).to_dict()
    _print_json(result)
    return 0


def _handle_gopro_prepare(args: argparse.Namespace) -> int:
    result = prepare_gopro_video(
        pattern=args.pattern,
        output_dir=args.output_dir,
        name=args.name,
        ffmpeg_bin=args.ffmpeg,
        crf=args.crf,
        preset=args.preset,
        video_codec=args.video_codec,
        audio_codec=args.audio_codec,
        fps=args.fps,
        scale=args.scale,
        recursive=args.recursive,
        transcode=not args.no_transcode,
    ).to_dict()
    _print_json(result)
    return 0


def _handle_collect(args: argparse.Namespace) -> int:
    settings, base_dir = _load_and_override_settings(args)
    result = collect_samples(
        source=args.source,
        mode=args.source_mode,
        settings=settings,
        base_dir=base_dir,
        output_dir=args.out_dir,
        limit_frames=args.limit_frames,
        calibrate_line=args.calibrate_line,
    )
    _print_json(result)
    return 0


def _handle_dataset_build(args: argparse.Namespace) -> int:
    result = build_dataset(
        raw_dir=args.raw_dir,
        dataset_dir=args.dataset_dir,
        train_ratio=args.train_ratio,
        seed=args.seed,
        clean=args.clean,
        include_unlabeled=args.include_unlabeled,
        classes=args.classes,
    ).to_dict()
    _print_json(result)
    return 0


def _handle_dataset_validate(args: argparse.Namespace) -> int:
    result = validate_dataset(args.dataset_dir).to_dict()
    _print_json(result)
    return 0


def _handle_train(args: argparse.Namespace) -> int:
    cfg = _load_script_config(getattr(args, "config", None))
    data_yaml = args.data_yaml or cfg.get("data_yaml")
    if not data_yaml:
        raise SystemExit("error: --data-yaml is required (or set 'data_yaml:' in config YAML)")
    result = train_model(
        data_yaml=data_yaml,
        model_path=args.model_path or cfg.get("model_path", "data/models/yolov8n.pt"),
        project_dir=args.project_dir or cfg.get("project_dir", "data/runs/detect"),
        run_name=args.run_name or cfg.get("run_name", "mx_plate_train"),
        epochs=args.epochs if args.epochs is not None else cfg.get("epochs", 100),
        imgsz=args.imgsz if args.imgsz is not None else cfg.get("imgsz", 640),
        batch=args.batch if args.batch is not None else cfg.get("batch", 16),
        device=args.device or cfg.get("device", "auto"),
        workers=args.workers if args.workers is not None else cfg.get("workers", 8),
    )
    _print_json(result)
    return 0


def _handle_detect_file(args: argparse.Namespace) -> int:
    cfg = _load_script_config(getattr(args, "config", None))
    source = args.source or cfg.get("source")
    if not source:
        raise SystemExit("error: --source is required (or set 'source:' in config YAML)")
    args.source = source
    out_dir = args.out_dir or cfg.get("out_dir")
    settings, base_dir = _load_and_override_settings(args)
    result = run_file_detection(
        source=source,
        settings=settings,
        base_dir=base_dir,
        output_dir=out_dir,
        limit_frames=args.limit_frames,
        calibrate_line=args.calibrate_line,
    )
    _print_json(result)
    return 0


def _handle_detect_stream(args: argparse.Namespace) -> int:
    cfg = _load_script_config(getattr(args, "config", None))
    source = args.source or cfg.get("source")
    if not source:
        raise SystemExit("error: --source is required (or set 'source:' in config YAML)")
    args.source = source
    out_dir = args.out_dir or cfg.get("out_dir")
    settings, base_dir = _load_and_override_settings(args)
    result = run_stream_detection(
        source=source,
        settings=settings,
        base_dir=base_dir,
        output_dir=out_dir,
        limit_frames=args.limit_frames,
        calibrate_line=args.calibrate_line,
    )
    _print_json(result)
    return 0


def _handle_reid_watch(args: argparse.Namespace) -> int:
    cfg = _load_script_config(getattr(args, "config", None))
    run_dir = args.run_dir or cfg.get("run_dir")
    if not run_dir:
        raise SystemExit("error: --run-dir is required (or set 'run_dir:' in config YAML)")
    run_reid_watch(
        run_dir=run_dir,
        device=args.device or cfg.get("device") or "cpu",
        threshold=args.threshold if args.threshold is not None else cfg.get("threshold", 0.60),
        poll_interval=args.poll_interval if args.poll_interval is not None else cfg.get("poll_interval", 2.0),
        plate_model_path=args.plate_model or cfg.get("plate_model"),
        plate_conf_low=args.plate_conf_low if args.plate_conf_low is not None else cfg.get("plate_conf_low", 0.15),
        stop_after_idle_sec=args.idle_timeout if args.idle_timeout is not None else cfg.get("idle_timeout"),
    )
    return 0


def _handle_recount(args: argparse.Namespace) -> int:
    cfg = _load_script_config(getattr(args, "config", None))
    run_dir = args.run_dir or cfg.get("run_dir")
    if not run_dir:
        raise SystemExit("error: --run-dir is required (or set 'run_dir:' in config YAML)")
    recount(run_dir=run_dir)
    return 0


def _handle_serve(args: argparse.Namespace) -> int:
    settings, _ = load_settings(args.config)
    host = args.host or settings.service.host
    port = args.port or settings.service.port
    run_service(host=host, port=port, default_config_path=args.config)
    return 0


def _add_detection_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="Path to YAML config (model settings + optional source/out_dir)")
    parser.add_argument("--source", help="Video file, stream URL or camera index (overrides config)")
    parser.add_argument("--out-dir", help="Directory for output artifacts")
    parser.add_argument("--limit-frames", type=int, help="Stop after N frames")
    parser.add_argument("--calibrate-line", action="store_true", help="Pick the finish line on the first frame")
    parser.add_argument("--line", help="Override finish line in x1,y1,x2,y2 or percent format")
    parser.add_argument("--line-width", type=int, help="Override finish line width in pixels")
    parser.add_argument("--vehicle-model", help="Override vehicle model path")
    parser.add_argument("--plate-model", help="Override plate model path")
    parser.add_argument("--device", help="Override runtime device (auto/cpu/cuda:0/mps)")
    parser.add_argument("--gallery", help="Override ReID gallery path")
    parser.add_argument("--enable-reid", action="store_true", help="Enable ReID fallback")
    parser.add_argument("--disable-reid", action="store_true", help="Disable ReID fallback")
    parser.add_argument("--digits-only", action="store_true", help="Treat plate model classes as 0..9 without a plate class")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mx-tracker", description="Motocross number plate training and detection pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    config_parser = subparsers.add_parser("config", help="Config helpers")
    config_sub = config_parser.add_subparsers(dest="config_command", required=True)
    config_init = config_sub.add_parser("init", help="Write default YAML config")
    config_init.add_argument("--output", default="configs/local.yaml", help="Destination path for the config")
    config_init.set_defaults(func=_handle_config_init)

    line_parser = subparsers.add_parser("line", help="Finish line helpers")
    line_sub = line_parser.add_subparsers(dest="line_command", required=True)
    line_calibrate = line_sub.add_parser("calibrate", help="Pick a finish line on the first frame")
    line_calibrate.add_argument("--source", required=True, help="Video file, stream URL or camera index")
    line_calibrate.set_defaults(func=_handle_line_calibrate)

    gopro_parser = subparsers.add_parser("gopro", help="GoPro preprocessing helpers")
    gopro_sub = gopro_parser.add_subparsers(dest="gopro_command", required=True)

    gopro_list = gopro_sub.add_parser("list", help="Build concat list in GoPro order")
    gopro_list.add_argument("--pattern", required=True, help="Glob pattern for GoPro files")
    gopro_list.add_argument("--output-list", default="files.txt", help="Output ffmpeg concat list")
    gopro_list.add_argument("--recursive", action="store_true", help="Enable recursive glob matching")
    gopro_list.set_defaults(func=_handle_gopro_list)

    gopro_prepare = gopro_sub.add_parser("prepare", help="Build GoPro concat list, merge files and optionally transcode")
    gopro_prepare.add_argument("--pattern", required=True, help="Glob pattern for GoPro files")
    gopro_prepare.add_argument("--output-dir", required=True, help="Directory for generated files")
    gopro_prepare.add_argument("--name", default="gopro", help="Base name for generated files")
    gopro_prepare.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg binary")
    gopro_prepare.add_argument("--crf", type=int, default=18, help="CRF for transcoding")
    gopro_prepare.add_argument("--preset", default="veryfast", help="ffmpeg preset")
    gopro_prepare.add_argument("--video-codec", default="libx264", help="Output video codec")
    gopro_prepare.add_argument("--audio-codec", default="copy", help="Output audio codec")
    gopro_prepare.add_argument("--fps", type=int, help="Optional output fps")
    gopro_prepare.add_argument("--scale", help="Optional ffmpeg scale, example 1920:1080")
    gopro_prepare.add_argument("--recursive", action="store_true", help="Enable recursive glob matching")
    gopro_prepare.add_argument("--no-transcode", action="store_true", help="Stop after concat merge without final H.264 transcode")
    gopro_prepare.set_defaults(func=_handle_gopro_prepare)

    collect_parser = subparsers.add_parser("collect", help="Collect bike crops for new training data")
    _add_detection_options(collect_parser)
    collect_parser.add_argument("--source-mode", choices=("file", "stream"), default="file", help="Read source as a finite file or incoming stream")
    collect_parser.add_argument("--loop-source", action="store_true", help="Loop local files in stream mode")
    collect_parser.add_argument("--reconnect-delay", type=float, help="Reconnect delay for stream mode")
    collect_parser.add_argument("--max-reconnects", type=int, help="Maximum reconnects for stream mode (-1 for unlimited)")
    collect_parser.set_defaults(func=_handle_collect)

    dataset_parser = subparsers.add_parser("dataset", help="Dataset build and validation")
    dataset_sub = dataset_parser.add_subparsers(dest="dataset_command", required=True)
    dataset_build = dataset_sub.add_parser("build", help="Create a clean YOLO dataset from raw crops")
    dataset_build.add_argument("--raw-dir", required=True, help="Directory with raw images and optional txt labels")
    dataset_build.add_argument("--dataset-dir", required=True, help="Output dataset directory")
    dataset_build.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio")
    dataset_build.add_argument("--seed", type=int, default=0, help="Random seed for splitting")
    dataset_build.add_argument("--clean", action="store_true", help="Delete existing dataset dir before rebuilding")
    dataset_build.add_argument("--include-unlabeled", action="store_true", help="Include unlabeled images as negative samples")
    dataset_build.add_argument("--classes", nargs="+", help="Override class names")
    dataset_build.set_defaults(func=_handle_dataset_build)

    dataset_validate = dataset_sub.add_parser("validate", help="Validate a prepared dataset")
    dataset_validate.add_argument("--dataset-dir", required=True, help="Dataset directory")
    dataset_validate.set_defaults(func=_handle_dataset_validate)

    train_parser = subparsers.add_parser("train", help="Train a YOLO plate+digits model")
    train_parser.add_argument("--config", default="configs/train.yaml", help="Path to train YAML config (sets all defaults)")
    train_parser.add_argument("--data-yaml", help="Path to dataset data.yaml (overrides config)")
    train_parser.add_argument("--model-path", help="Base model checkpoint (overrides config)")
    train_parser.add_argument("--project-dir", help="Output project directory (overrides config)")
    train_parser.add_argument("--run-name", help="Training run name (overrides config)")
    train_parser.add_argument("--epochs", type=int, default=None, help="Training epochs (default: 100)")
    train_parser.add_argument("--imgsz", type=int, default=None, help="Training image size (default: 640)")
    train_parser.add_argument("--batch", type=int, default=None, help="Batch size (default: 16)")
    train_parser.add_argument("--device", help="Training device (default: auto)")
    train_parser.add_argument("--workers", type=int, default=None, help="Data loader workers (default: 8)")
    train_parser.set_defaults(func=_handle_train)

    detect_parser = subparsers.add_parser("detect", help="Detection modes")
    detect_sub = detect_parser.add_subparsers(dest="detect_command", required=True)

    detect_file = detect_sub.add_parser("file", help="Process a finite video file")
    _add_detection_options(detect_file)
    detect_file.set_defaults(func=_handle_detect_file)

    detect_stream = detect_sub.add_parser("stream", help="Process a live source or simulated stream")
    _add_detection_options(detect_stream)
    detect_stream.add_argument("--loop-source", action="store_true", help="Loop local files instead of stopping at EOF")
    detect_stream.add_argument("--reconnect-delay", type=float, help="Reconnect delay for stream mode")
    detect_stream.add_argument("--max-reconnects", type=int, help="Maximum reconnects for stream mode (-1 for unlimited)")
    detect_stream.set_defaults(func=_handle_detect_stream)

    reid_watch_parser = subparsers.add_parser(
        "reid-watch",
        help="Watch an active run directory and match unresolved crossings via ReID",
    )
    reid_watch_parser.add_argument("--config", default="configs/reid_watch.yaml", help="Path to reid_watch YAML config (sets all defaults)")
    reid_watch_parser.add_argument("--run-dir", help="Run directory produced by detect (overrides config)")
    reid_watch_parser.add_argument("--device", help="Device for plate model and ReID (cpu/cuda:0/mps)")
    reid_watch_parser.add_argument("--threshold", type=float, default=None, help="ReID similarity threshold (0–1, default: 0.60)")
    reid_watch_parser.add_argument("--poll-interval", type=float, default=None, help="Seconds between polls (default: 2.0)")
    reid_watch_parser.add_argument("--plate-model", help="Plate model path for re-detection (tried before ReID)")
    reid_watch_parser.add_argument("--plate-conf-low", type=float, default=None, help="Lower confidence threshold for plate re-detection (default: 0.15)")
    reid_watch_parser.add_argument("--idle-timeout", type=float, default=None, help="Stop after N idle seconds (default: run until Ctrl-C)")
    reid_watch_parser.set_defaults(func=_handle_reid_watch)

    recount_parser = subparsers.add_parser(
        "recount",
        help="Recount laps from events.jsonl and write results.csv",
    )
    recount_parser.add_argument("--config", default="configs/recount.yaml", help="Path to recount YAML config")
    recount_parser.add_argument("--run-dir", help="Run directory produced by detect (overrides config)")
    recount_parser.set_defaults(func=_handle_recount)

    serve_parser = subparsers.add_parser("serve", help="Run the HTTP job service")
    serve_parser.add_argument("--config", help="Default YAML config used by service jobs")
    serve_parser.add_argument("--host", help="Bind host")
    serve_parser.add_argument("--port", type=int, help="Bind port")
    serve_parser.set_defaults(func=_handle_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))
