from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

from .runtime import REPO_ROOT


DEFAULT_CONFIG: dict[str, Any] = {
    "runtime": {
        "device": "auto",
        "source_fps_fallback": 30.0,
    },
    "line": {
        "value": "50%,5%,50%,95%",
        "width": 24,
        "cooldown_sec": 2.0,
        "direction": "either",
        "read_distance_multiplier": 2.5,
    },
    "models": {
        "vehicle_model": "yolov8n.pt",
        "plate_model": "runs/detect/train3/weights/best.pt",
        "tracker": "botsort.yaml",
        "motorcycle_class_id": 3,
        "vehicle_conf": 0.35,
        "vehicle_iou": 0.50,
        "vehicle_imgsz": 1280,
        "plate_conf": 0.25,
        "plate_has_class": True,
        "plate_class_id": 0,
        "bike_crop_expand": 1.10,
        "plate_box_expand": 0.12,
    },
    "reads": {
        "scan_every_n_frames": 1,
        "min_digits": 1,
        "vote_window_sec": 2.5,
        "track_state_ttl_sec": 10.0,
    },
    "reid": {
        "enabled": False,
        "gallery_path": "gallery",
        "threshold": 0.60,
    },
    "stream": {
        "reconnect_delay_sec": 3.0,
        "max_reconnects": -1,
        "loop_file": False,
        "status_interval_sec": 10.0,
    },
    "output": {
        "write_video": True,
        "write_csv": True,
        "write_jsonl": True,
        "write_summary": True,
        "overlay_top_n": 10,
        "save_plate_crops": False,
    },
    "service": {
        "host": "127.0.0.1",
        "port": 8080,
    },
}


def _deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def resolve_path(base_dir: Path, value: str | None) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


class RuntimeSettings(BaseModel):
    device: str = "auto"
    source_fps_fallback: float = 30.0


class LineSettings(BaseModel):
    value: str = "50%,5%,50%,95%"
    width: int = 24
    cooldown_sec: float = 2.0
    direction: str = "either"
    read_distance_multiplier: float = 2.5

    @field_validator("direction")
    @classmethod
    def validate_direction(cls, value: str) -> str:
        allowed = {"either", "positive", "negative"}
        if value not in allowed:
            raise ValueError(f"line.direction must be one of {sorted(allowed)}")
        return value


class ModelSettings(BaseModel):
    vehicle_model: str = "yolov8n.pt"
    plate_model: str = "runs/detect/train3/weights/best.pt"
    tracker: str = "botsort.yaml"
    motorcycle_class_id: int = 3
    vehicle_conf: float = 0.35
    vehicle_iou: float = 0.50
    vehicle_imgsz: int = 1280
    plate_conf: float = 0.25
    plate_has_class: bool = True
    plate_class_id: int = 0
    bike_crop_expand: float = 1.10
    plate_box_expand: float = 0.12


class ReadSettings(BaseModel):
    scan_every_n_frames: int = 1
    min_digits: int = 1
    vote_window_sec: float = 2.5
    track_state_ttl_sec: float = 10.0


class ReIdSettings(BaseModel):
    enabled: bool = False
    gallery_path: str = "gallery"
    threshold: float = 0.60


class StreamSettings(BaseModel):
    reconnect_delay_sec: float = 3.0
    max_reconnects: int = -1
    loop_file: bool = False
    status_interval_sec: float = 10.0


class OutputSettings(BaseModel):
    write_video: bool = True
    write_csv: bool = True
    write_jsonl: bool = True
    write_summary: bool = True
    overlay_top_n: int = 10
    save_plate_crops: bool = False


class ServiceSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8080


class TrackerSettings(BaseModel):
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    line: LineSettings = Field(default_factory=LineSettings)
    models: ModelSettings = Field(default_factory=ModelSettings)
    reads: ReadSettings = Field(default_factory=ReadSettings)
    reid: ReIdSettings = Field(default_factory=ReIdSettings)
    stream: StreamSettings = Field(default_factory=StreamSettings)
    output: OutputSettings = Field(default_factory=OutputSettings)
    service: ServiceSettings = Field(default_factory=ServiceSettings)


def load_settings(config_path: str | Path | None = None) -> tuple[TrackerSettings, Path]:
    data = deepcopy(DEFAULT_CONFIG)
    base_dir = REPO_ROOT
    if config_path is not None:
        path = Path(config_path).expanduser().resolve()
        loaded = yaml.safe_load(path.read_text()) or {}
        if not isinstance(loaded, dict):
            raise ValueError("Config root must be a mapping")
        _deep_merge(data, loaded)
        base_dir = path.parent
    settings = TrackerSettings.model_validate(data)
    return settings, base_dir


def write_default_config(destination: str | Path) -> Path:
    path = Path(destination).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(DEFAULT_CONFIG, sort_keys=False))
    return path
