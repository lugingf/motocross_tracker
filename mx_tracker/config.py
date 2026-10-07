from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .runtime import REPO_ROOT


DEFAULT_CONFIG: dict[str, Any] = {
    "runtime": {
        "device": "auto",
        "source_fps_fallback": 30.0,
        "profile": False,
    },
    "line": {
        "value": "50%,5%,50%,95%",
        "width": 24,
        "cooldown_sec": 2.0,
        "direction": "left_to_right",
        "read_distance_multiplier": 2.5,
    },
    "models": {
        "vehicle_model": "data/models/yolov8n.pt",
        "plate_model": "data/runs/detect/train3/weights/best.pt",
        "tracker": "botsort.yaml",
        "motorcycle_class_id": 3,
        "vehicle_conf": 0.35,
        "vehicle_iou": 0.50,
        "vehicle_imgsz": 1280,
        "plate_conf": 0.25,
        "plate_has_class": True,
        "plate_class_id": 0,
        "bike_crop_expand": 0.10,
        "plate_box_expand": 0.12,
        "min_bike_crop_px": 96,
        "plate_zone_n": 1,
        "plate_zone_select": [1],
    },
    "motion_gate": {
        "enabled": True,
        "zone_fraction": 0.25,
        "downscale": 4,
        "pixel_delta": 25.0,
        "min_area_fraction": 0.004,
        "hold_sec": 1.0,
        "warmup_frames": 30,
        "background_alpha": 0.02,
    },
    "reads": {
        "scan_every_n_frames": 1,
        "min_digits": 1,
        "vote_window_sec": 2.5,
        "track_state_ttl_sec": 10.0,
    },
    "reid": {
        "enabled": False,
        "gallery_path": "data/gallery",
        "threshold": 0.60,
    },
    "stream": {
        "reconnect_delay_sec": 3.0,
        "max_reconnects": -1,
        "loop_file": False,
        "status_interval_sec": 10.0,
        "ingest": "pyav",
        "open_timeout_sec": 10.0,
        "read_timeout_sec": 5.0,
        "rtsp_transport": "tcp",
        "rotation": 0,
    },
    "output": {
        "write_video": True,
        "write_csv": True,
        "write_jsonl": True,
        "write_summary": True,
        "overlay_top_n": 10,
        "save_plate_crops": False,
        "overlay_crossing_text": False,
        "save_bike_crops_unresolved": False,
    },
    "service": {
        "host": "127.0.0.1",
        "port": 8080,
    },
    "integration": {
        "enabled": False,
        "host": "127.0.0.1",
        "port": 8081,
        "token": "",
        "state_dir": "data/integration",
        "allowed_sources": [],
        "allowed_callbacks": [],
        "max_jobs": 1,
        "batch_size": 200,
        "heartbeat_interval_sec": 10.0,
        "revision_interval_sec": 30.0,
        "low_confidence": 0.5,
        "max_outbox_events": 100000,
        "request_timeout_sec": 15.0,
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
    profile: bool = False


class LineSettings(BaseModel):
    model_config = ConfigDict(validate_default=True)

    value: str = "50%,5%,50%,95%"
    width: int = 24
    cooldown_sec: float = 2.0
    direction: str = "left_to_right"
    read_distance_multiplier: float = 2.5

    @field_validator("direction")
    @classmethod
    def validate_direction(cls, value: str) -> str:
        allowed = {"left_to_right", "right_to_left"}
        if value not in allowed:
            raise ValueError(f"line.direction must be one of {sorted(allowed)}")
        return value


class ModelSettings(BaseModel):
    vehicle_model: str = "data/models/yolov8n.pt"
    plate_model: str = "data/runs/detect/train3/weights/best.pt"
    tracker: str = "botsort.yaml"
    motorcycle_class_id: int = 3
    vehicle_conf: float = 0.35
    vehicle_iou: float = 0.50
    vehicle_imgsz: int = 1280
    plate_conf: float = 0.25
    plate_has_class: bool = True
    plate_class_id: int = 0
    bike_crop_expand: float = 0.10
    plate_box_expand: float = 0.12
    min_bike_crop_px: int = 96
    plate_zone_n: int = 1
    plate_zone_select: list[int] = Field(default_factory=lambda: [1])

    @field_validator("plate_zone_n")
    @classmethod
    def validate_plate_zone_n(cls, v: int) -> int:
        allowed = {1, 2, 4, 9, 32}
        if v not in allowed:
            raise ValueError(f"plate_zone_n must be one of {sorted(allowed)}")
        return v

    @model_validator(mode="after")
    def validate_plate_zone_select(self) -> "ModelSettings":
        out_of_range = [z for z in self.plate_zone_select if not 1 <= z <= self.plate_zone_n]
        if out_of_range:
            raise ValueError(
                f"plate_zone_select entries {out_of_range} are outside 1..{self.plate_zone_n} "
                f"(plate_zone_n={self.plate_zone_n})"
            )
        return self


class MotionGateSettings(BaseModel):
    """Skip detection on frames where nothing has entered the frame yet."""

    model_config = ConfigDict(validate_default=True)

    enabled: bool = True
    zone_fraction: float = Field(default=0.25, gt=0.0, le=1.0)
    downscale: int = Field(default=4, ge=1)
    pixel_delta: float = Field(default=25.0, gt=0.0, le=255.0)
    min_area_fraction: float = Field(default=0.004, ge=0.0, le=1.0)
    hold_sec: float = Field(default=1.0, ge=0.0)
    warmup_frames: int = Field(default=30, ge=0)
    background_alpha: float = Field(default=0.02, gt=0.0, le=1.0)


class ReadSettings(BaseModel):
    scan_every_n_frames: int = 1
    min_digits: int = 1
    vote_window_sec: float = 2.5
    track_state_ttl_sec: float = 10.0


class ReIdSettings(BaseModel):
    enabled: bool = False
    gallery_path: str = "data/gallery"
    threshold: float = 0.60


class StreamSettings(BaseModel):
    reconnect_delay_sec: float = 3.0
    max_reconnects: int = -1
    loop_file: bool = False
    status_interval_sec: float = 10.0
    # How network streams are read. "pyav" takes the timeline from the media's own
    # timestamps; "cv2" stamps frames with the wall clock at the moment they were
    # decoded, which drifts whenever processing falls behind the stream.
    ingest: str = "pyav"
    open_timeout_sec: float = Field(default=10.0, gt=0.0)
    read_timeout_sec: float = Field(default=5.0, gt=0.0)
    rtsp_transport: str = "tcp"
    # Degrees clockwise the frame is turned to stand upright, as a phone held
    # sideways delivers it. Everything after the source sees the turned frame.
    rotation: int = 0

    @field_validator("rotation")
    @classmethod
    def validate_rotation(cls, value: int) -> int:
        if value not in (0, 90, 180, 270):
            raise ValueError("stream.rotation must be 0, 90, 180 or 270")
        return value

    @field_validator("ingest")
    @classmethod
    def validate_ingest(cls, value: str) -> str:
        allowed = {"pyav", "cv2"}
        if value not in allowed:
            raise ValueError(f"stream.ingest must be one of {sorted(allowed)}")
        return value


class OutputSettings(BaseModel):
    write_video: bool = True
    write_csv: bool = True
    write_jsonl: bool = True
    write_summary: bool = True
    overlay_top_n: int = 10
    save_plate_crops: bool = False
    overlay_crossing_text: bool = False
    save_bike_crops_unresolved: bool = False


class ServiceSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8080


class IntegrationSettings(BaseModel):
    """The API lap_vision drives the tracker through, served on a listener of its own.

    The listener is separate from `service` on purpose: it is the one that gets
    forwarded to another machine, and it must not carry the job service's
    routes, which take arbitrary paths and have no authentication.
    """

    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 8081
    # Better supplied as MX_INTEGRATION_TOKEN than written in a file.
    token: str = ""
    state_dir: str = "data/integration"
    # URL prefixes the API may read streams from and post results to. Empty
    # means none: a request naming anything else is refused.
    allowed_sources: list[str] = Field(default_factory=list)
    allowed_callbacks: list[str] = Field(default_factory=list)
    max_jobs: int = Field(default=1, ge=1)
    batch_size: int = Field(default=200, ge=1, le=500)
    heartbeat_interval_sec: float = Field(default=10.0, gt=0.0)
    revision_interval_sec: float = Field(default=30.0, gt=0.0)
    low_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    max_outbox_events: int = Field(default=100000, ge=1)
    request_timeout_sec: float = Field(default=15.0, gt=0.0)


class TrackerSettings(BaseModel):
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    line: LineSettings = Field(default_factory=LineSettings)
    models: ModelSettings = Field(default_factory=ModelSettings)
    motion_gate: MotionGateSettings = Field(default_factory=MotionGateSettings)
    reads: ReadSettings = Field(default_factory=ReadSettings)
    reid: ReIdSettings = Field(default_factory=ReIdSettings)
    stream: StreamSettings = Field(default_factory=StreamSettings)
    output: OutputSettings = Field(default_factory=OutputSettings)
    service: ServiceSettings = Field(default_factory=ServiceSettings)
    integration: IntegrationSettings = Field(default_factory=IntegrationSettings)


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
