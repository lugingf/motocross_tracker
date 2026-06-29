"""Loading and ownership of the YOLO models used by detection.

`DetectionModels.load()` is the single place that resolves the runtime device,
locates weights relative to the config/repo, and instantiates the vehicle
tracker, plate reader and (optionally) the ReID matcher. Everything else in
the pipeline receives an already-built bundle.
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import torch

from .config import TrackerSettings, resolve_path
from .plate_reading import PlateReader
from .runtime import REPO_ROOT, prepare_runtime_environment

prepare_runtime_environment()

from ultralytics import YOLO

if TYPE_CHECKING:
    from .reid import ReIdentifier

Logger = Callable[[str], None]


def get_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_runtime_path(base_dir: Path, value: str | None) -> Path | None:
    """Resolve a path against the config dir, falling back to the repo root."""
    path = resolve_path(base_dir, value)
    if path is not None and path.exists():
        return path
    return resolve_path(REPO_ROOT, value)


def resolve_tracker_path(value: str, base_dir: Path) -> str:
    explicit = resolve_runtime_path(base_dir, value)
    if explicit is not None and explicit.exists():
        return str(explicit)
    spec = importlib.util.find_spec("ultralytics")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("ultralytics package not found")
    tracker_path = Path(spec.submodule_search_locations[0]) / "cfg" / "trackers" / value
    if not tracker_path.exists():
        raise FileNotFoundError(f"Tracker config not found: {value}")
    return str(tracker_path)


def _load_reid(settings: TrackerSettings, base_dir: Path, logger: Logger) -> "ReIdentifier | None":
    if not settings.reid.enabled:
        return None
    from .reid import ReIdentifier

    gallery_path = resolve_runtime_path(base_dir, settings.reid.gallery_path)
    if gallery_path is None:
        return None
    logger(f"reid_gallery={gallery_path}")
    return ReIdentifier(str(gallery_path), device=settings.runtime.device, thresh=settings.reid.threshold)


@dataclass
class DetectionModels:
    """The loaded models and runtime parameters shared across a run."""

    vehicle_model: YOLO
    plate_reader: PlateReader
    reid: "ReIdentifier | None"
    device: str
    tracker_yaml: str

    @classmethod
    def load(
        cls,
        settings: TrackerSettings,
        base_dir: Path,
        collect_only: bool,
        logger: Logger,
    ) -> "DetectionModels":
        device = get_device(settings.runtime.device)
        vehicle_model_path = resolve_runtime_path(base_dir, settings.models.vehicle_model)
        if vehicle_model_path is None or not vehicle_model_path.exists():
            raise FileNotFoundError(f"Vehicle model not found: {settings.models.vehicle_model}")
        vehicle_model = YOLO(str(vehicle_model_path))
        vehicle_model.to(device)

        plate_model = None
        plate_model_path = resolve_runtime_path(base_dir, settings.models.plate_model)
        if not collect_only:
            if plate_model_path is not None and plate_model_path.exists():
                plate_model = YOLO(str(plate_model_path))
                plate_model.to(device)
            else:
                raise FileNotFoundError(f"Plate model not found: {settings.models.plate_model}")

        logger(f"device={device}")
        logger(f"vehicle_model={vehicle_model_path}")
        if plate_model_path is not None:
            logger(f"plate_model={plate_model_path}")

        return cls(
            vehicle_model=vehicle_model,
            plate_reader=PlateReader(plate_model, settings),
            reid=None if collect_only else _load_reid(settings, base_dir, logger),
            device=device,
            tracker_yaml=resolve_tracker_path(settings.models.tracker, base_dir),
        )

    def track(self, frame, settings: TrackerSettings):
        """Run the vehicle tracker on one frame, returning the first result."""
        return self.vehicle_model.track(
            frame,
            tracker=self.tracker_yaml,
            persist=True,
            device=self.device,
            conf=settings.models.vehicle_conf,
            iou=settings.models.vehicle_iou,
            imgsz=settings.models.vehicle_imgsz,
            verbose=False,
        )[0]
