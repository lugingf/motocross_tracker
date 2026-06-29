"""Run output: directory layout, event logs and saved crops.

This is the persistence layer. `RunArtifacts` describes where things go,
`EventLog` is the single writer for crossing/collect events (kills the
duplicated event-dict construction the pipeline used to carry), and
`CropArchive` owns the resolved/unresolved crop folders and sidecars.
"""
from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

from .plate_reading import PlateRead
from .runtime import REPO_ROOT
from .tracking import TrackState

# PlateRead/TrackState are used for type hints in CropArchive; importing them
# keeps the sidecar metadata format colocated with the data it serializes.

COLLECT_CSV_FIELDS = [
    "timestamp",
    "frame_index",
    "tracker_id",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "center_x",
    "center_y",
    "crop_path",
]

EVENT_CSV_FIELDS = [
    "timestamp",
    "wall_time",
    "frame_index",
    "tracker_id",
    "rider_id",
    "identity_source",
    "plate_text",
    "plate_conf",
    "lap",
    "lap_time",
    "center_x",
    "center_y",
    "crop_file",
]


@dataclass(slots=True)
class RunArtifacts:
    run_dir: Path
    video_path: Path | None
    csv_path: Path | None
    jsonl_path: Path | None
    summary_path: Path | None
    debug_dir: Path | None


class JsonlWriter:
    def __init__(self, path: Path | None) -> None:
        self.path = path
        self.handle = None if path is None else path.open("w", encoding="utf-8")

    def write(self, payload: dict[str, object]) -> None:
        if self.handle is None:
            return
        self.handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.handle.flush()

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()


class CsvWriter:
    def __init__(self, path: Path | None, fieldnames: list[str]) -> None:
        self.path = path
        self.handle = None if path is None else path.open("w", newline="", encoding="utf-8")
        self.writer = None if self.handle is None else csv.DictWriter(self.handle, fieldnames=fieldnames, extrasaction="ignore")
        if self.writer is not None:
            self.writer.writeheader()
            self.handle.flush()

    def write(self, payload: dict[str, object]) -> None:
        if self.writer is None:
            return
        self.writer.writerow(payload)
        self.handle.flush()

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()


def _flatten_event(event: dict[str, object]) -> dict[str, object]:
    bbox = event.pop("bbox")
    center = event.pop("center")
    x1, y1, x2, y2 = bbox
    cx, cy = center
    event.update(
        {
            "bbox_x1": x1,
            "bbox_y1": y1,
            "bbox_x2": x2,
            "bbox_y2": y2,
            "center_x": cx,
            "center_y": cy,
        }
    )
    return event


def _flatten_collect_row(
    timestamp: float,
    frame_index: int,
    tracker_id: int,
    bbox: tuple[int, int, int, int],
    center: tuple[int, int],
    crop_path: Path,
) -> dict[str, object]:
    return {
        "timestamp": round(timestamp, 3),
        "frame_index": frame_index,
        "tracker_id": tracker_id,
        "bbox_x1": bbox[0],
        "bbox_y1": bbox[1],
        "bbox_x2": bbox[2],
        "bbox_y2": bbox[3],
        "center_x": center[0],
        "center_y": center[1],
        "crop_path": str(crop_path),
    }


def _make_run_dir(output_dir: str | Path | None, prefix: str) -> Path:
    if output_dir is not None:
        run_dir = Path(output_dir).expanduser().resolve()
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = (REPO_ROOT / "data" / "artifacts" / f"{prefix}_{timestamp}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def prepare_artifacts(
    output_dir: str | Path | None,
    prefix: str,
    write_video: bool,
    write_csv: bool,
    write_jsonl: bool,
    write_summary: bool,
    save_plate_crops: bool,
) -> RunArtifacts:
    run_dir = _make_run_dir(output_dir, prefix)
    debug_dir = run_dir / "plate_crops" if save_plate_crops else None
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
    return RunArtifacts(
        run_dir=run_dir,
        video_path=run_dir / "overlay.mp4" if write_video else None,
        csv_path=run_dir / "events.csv" if write_csv else None,
        jsonl_path=run_dir / "events.jsonl" if write_jsonl else None,
        summary_path=run_dir / "summary.json" if write_summary else None,
        debug_dir=debug_dir,
    )


class EventLog:
    """Single sink for run events, writing to both JSONL and CSV.

    Owns the wall-clock origin so every event gets a consistent `wall_time`,
    and centralizes the event schema that used to be repeated per code path.
    """

    def __init__(self, artifacts: RunArtifacts, wall_started_at: datetime, collect_only: bool) -> None:
        fields = COLLECT_CSV_FIELDS if collect_only else EVENT_CSV_FIELDS
        self._jsonl = JsonlWriter(artifacts.jsonl_path)
        self._csv = CsvWriter(artifacts.csv_path, fields)
        self._wall_started_at = wall_started_at

    def emit_crossing(
        self,
        *,
        timestamp: float,
        frame_index: int,
        tracker_id: int,
        rider_id: str,
        identity_source: str,
        plate_text: str,
        plate_conf: float,
        bbox: tuple[int, int, int, int],
        center: tuple[int, int],
        crop_file: str,
    ) -> None:
        event: dict[str, object] = {
            "timestamp": round(timestamp, 3),
            "wall_time": (self._wall_started_at + timedelta(seconds=timestamp)).isoformat(timespec="seconds"),
            "frame_index": frame_index,
            "tracker_id": tracker_id,
            "rider_id": rider_id,
            "identity_source": identity_source,
            "plate_text": plate_text,
            "plate_conf": round(plate_conf, 3),
            "lap": "",
            "lap_time": "",
            "bbox": [bbox[0], bbox[1], bbox[2], bbox[3]],
            "center": [center[0], center[1]],
            "crop_file": crop_file,
        }
        self._jsonl.write(event)
        self._csv.write(_flatten_event(dict(event)))

    def emit_collect(
        self,
        *,
        timestamp: float,
        frame_index: int,
        tracker_id: int,
        bbox: tuple[int, int, int, int],
        center: tuple[int, int],
        crop_path: Path,
    ) -> None:
        self._jsonl.write(
            {
                "timestamp": round(timestamp, 3),
                "frame_index": frame_index,
                "tracker_id": tracker_id,
                "bbox": [bbox[0], bbox[1], bbox[2], bbox[3]],
                "center": [center[0], center[1]],
                "crop_path": str(crop_path),
            }
        )
        self._csv.write(_flatten_collect_row(timestamp, frame_index, tracker_id, bbox, center, crop_path))

    def close(self) -> None:
        self._csv.close()
        self._jsonl.close()


class CropArchive:
    """Saves plate crops into resolved/<plate> or unresolved/, with sidecars."""

    def __init__(self, debug_dir: Path | None) -> None:
        self.debug_dir = debug_dir

    @property
    def enabled(self) -> bool:
        return self.debug_dir is not None

    def save(
        self,
        frame_index: int,
        tracker_id: int,
        crop: np.ndarray,
        plate_text: str | None = None,
        track_state: TrackState | None = None,
        plate_read: PlateRead | None = None,
        bike_crop: np.ndarray | None = None,
    ) -> Path | None:
        if self.debug_dir is None or crop.size == 0:
            return None
        if plate_text:
            dest = self.debug_dir / "resolved" / f"plate_{plate_text}"
        else:
            dest = self.debug_dir / "unresolved"
        dest.mkdir(parents=True, exist_ok=True)
        stem = f"frame{frame_index}_tid{tracker_id}"
        output_path = dest / f"{stem}.jpg"
        cv2.imwrite(str(output_path), crop.copy())
        if plate_text is None and bike_crop is not None and bike_crop.size > 0:
            cv2.imwrite(str(dest / f"{stem}_bike.jpg"), bike_crop)

        meta: dict[str, object] = {
            "frame_index": frame_index,
            "tracker_id": tracker_id,
            "plate_text": plate_text,
        }
        if plate_read is not None:
            meta["last_read"] = {
                "text": plate_read.text,
                "confidence": round(plate_read.confidence, 3) if plate_read.confidence else None,
            }
        if track_state is not None:
            meta["observations"] = [
                {
                    "text": obs.text,
                    "confidence": round(obs.confidence, 3),
                    "frame_index": obs.frame_index,
                }
                for obs in track_state.observations
            ]
        if not plate_text:
            meta["manual_plate"] = ""
            meta["save"] = 0
        (dest / f"{stem}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        return output_path
