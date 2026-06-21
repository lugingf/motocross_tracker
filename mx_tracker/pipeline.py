from __future__ import annotations

import csv
import importlib.util
import json
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from itertools import chain
from pathlib import Path
from threading import Event
from typing import Callable, Iterator

import cv2
import numpy as np
import torch

from .config import TrackerSettings, resolve_path
from .geometry import (
    CrossingDecision,
    detect_crossing,
    expand_bbox,
    parse_line_arg,
    pick_line_on_frame,
    point_line_side_and_dist,
    to_percent_str,
    zone_crop,
)
from .runtime import REPO_ROOT, prepare_runtime_environment

prepare_runtime_environment()

from ultralytics import YOLO


Logger = Callable[[str], None]


@dataclass(slots=True)
class PlateObservation:
    text: str
    confidence: float
    timestamp: float
    frame_index: int


@dataclass(slots=True)
class PlateRead:
    text: str | None
    confidence: float
    plate_bbox: tuple[int, int, int, int] | None
    x_center: float = 0.0


@dataclass(slots=True)
class TrackState:
    last_center: tuple[int, int] | None = None
    last_cross_ts: float = -1e9
    last_seen_ts: float = 0.0
    observations: deque[PlateObservation] = field(default_factory=deque)
    stable_plate: PlateObservation | None = None

    def add_observation(self, observation: PlateObservation, ttl_sec: float) -> None:
        self.observations.append(observation)
        self.stable_plate = observation
        self.prune(observation.timestamp, ttl_sec)

    def prune(self, current_ts: float, ttl_sec: float) -> None:
        while self.observations and (current_ts - self.observations[0].timestamp) > ttl_sec:
            self.observations.popleft()
        if self.stable_plate is not None and (current_ts - self.stable_plate.timestamp) > ttl_sec:
            self.stable_plate = None

    def best_plate(self, current_ts: float, vote_window_sec: float) -> PlateObservation | None:
        score_by_text: dict[str, float] = defaultdict(float)
        count_by_text: dict[str, int] = defaultdict(int)
        best_by_text: dict[str, PlateObservation] = {}
        for observation in self.observations:
            if (current_ts - observation.timestamp) > vote_window_sec:
                continue
            score_by_text[observation.text] += observation.confidence
            count_by_text[observation.text] += 1
            previous = best_by_text.get(observation.text)
            if previous is None or observation.confidence > previous.confidence:
                best_by_text[observation.text] = observation
        if not best_by_text:
            return self.stable_plate
        return max(
            best_by_text.values(),
            key=lambda obs: (
                count_by_text[obs.text],
                score_by_text[obs.text],
                obs.confidence,
                len(obs.text),
            ),
        )


@dataclass(slots=True)
class DetectionRecord:
    tracker_id: int
    bbox: tuple[int, int, int, int]
    center: tuple[int, int]
    plate_text: str | None = None
    plate_conf: float = 0.0


@dataclass(slots=True)
class FramePacket:
    frame: np.ndarray
    frame_index: int
    timestamp: float
    fps: float
    width: int
    height: int


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
        self.writer = None if self.handle is None else csv.DictWriter(self.handle, fieldnames=fieldnames)
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


def _default_logger(message: str) -> None:
    print(message)


def get_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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


def resolve_runtime_path(base_dir: Path, value: str | None) -> Path | None:
    path = resolve_path(base_dir, value)
    if path is not None and path.exists():
        return path
    fallback = resolve_path(REPO_ROOT, value)
    return fallback


def _is_numeric_source(value: str) -> bool:
    return value.isdigit()


def _is_local_file_source(value: str) -> bool:
    if _is_numeric_source(value):
        return False
    return Path(value).expanduser().exists()


def _open_capture(source: str) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(int(source) if _is_numeric_source(source) else source)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")
    return capture


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


def _xyxy_array(result_boxes: object, attr: str) -> np.ndarray:
    tensor = getattr(result_boxes, attr)
    return tensor.detach().cpu().numpy()


def _build_digit_groups(
    plate_model: YOLO | None,
    crop: np.ndarray,
    settings: TrackerSettings,
) -> tuple[list[list[dict]], tuple[int, int, int, int] | None]:
    if plate_model is None or crop.size == 0:
        return [], None
    result = plate_model(crop, verbose=False)[0]
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return [], None

    xyxy = _xyxy_array(boxes, "xyxy")
    classes = _xyxy_array(boxes, "cls").astype(int)
    confidences = _xyxy_array(boxes, "conf")

    plate_candidates: list[tuple[float, tuple[int, int, int, int]]] = []
    digit_candidates: list[dict[str, float | str]] = []
    for raw_bbox, class_id, confidence in zip(xyxy, classes, confidences):
        conf = float(confidence)
        if conf < settings.models.plate_conf:
            continue
        x1, y1, x2, y2 = map(int, raw_bbox.tolist())
        if settings.models.plate_has_class and class_id == settings.models.plate_class_id:
            plate_candidates.append((conf, (x1, y1, x2, y2)))
            continue
        digit = class_id - 1 if settings.models.plate_has_class else class_id
        if not (0 <= digit <= 9):
            continue
        digit_candidates.append(
            {
                "x_center": float(x1 + x2) / 2.0,
                "width": float(max(1, x2 - x1)),
                "digit": str(digit),
                "conf": conf,
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
            }
        )

    plate_bbox = None
    if plate_candidates:
        _, plate_bbox = max(plate_candidates, key=lambda item: item[0])

    if plate_bbox is not None:
        px1, py1, px2, py2 = expand_bbox(
            plate_bbox[0],
            plate_bbox[1],
            plate_bbox[2],
            plate_bbox[3],
            crop.shape[1],
            crop.shape[0],
            scale=1.0 + settings.models.plate_box_expand,
        )
        filtered_digits = []
        for candidate in digit_candidates:
            cx = float(candidate["x_center"])
            cy = float(candidate["y1"] + candidate["y2"]) / 2.0
            if px1 <= cx <= px2 and py1 <= cy <= py2:
                filtered_digits.append(candidate)
        if filtered_digits:
            digit_candidates = filtered_digits

    if not digit_candidates:
        return [], plate_bbox

    digit_candidates.sort(key=lambda item: item["x_center"])
    deduped: list[dict[str, float | str]] = []
    for candidate in digit_candidates:
        if not deduped:
            deduped.append(candidate)
            continue
        prev = deduped[-1]
        threshold = 0.35 * max(float(prev["width"]), float(candidate["width"]))
        if abs(float(candidate["x_center"]) - float(prev["x_center"])) <= threshold:
            if float(candidate["conf"]) > float(prev["conf"]):
                deduped[-1] = candidate
            continue
        deduped.append(candidate)

    # Split into groups when gap between digits exceeds one digit width —
    # prevents merging digits from two bikes in the same crop.
    groups: list[list[dict]] = []
    for candidate in deduped:
        if not groups:
            groups.append([candidate])
            continue
        prev = groups[-1][-1]
        gap = float(candidate["x1"]) - float(prev["x2"])
        avg_width = (float(prev["width"]) + float(candidate["width"])) / 2.0
        if gap > avg_width:
            groups.append([candidate])
        else:
            groups[-1].append(candidate)

    return groups, plate_bbox


def _group_score(g: list[dict]) -> tuple[int, float]:
    avg_conf = sum(float(d["conf"]) for d in g) / len(g)
    return (len(g), avg_conf)


def _group_to_plate_read(
    group: list[dict],
    plate_bbox: tuple[int, int, int, int] | None,
    min_digits: int,
) -> PlateRead | None:
    number = "".join(str(item["digit"]) for item in group)
    if len(number) < min_digits:
        return None
    confidence = sum(float(item["conf"]) for item in group) / len(group)
    x_center = sum(float(d["x_center"]) for d in group) / len(group)
    return PlateRead(number, confidence, plate_bbox, x_center=x_center)


def detect_plate_number(
    plate_model: YOLO | None,
    crop: np.ndarray,
    settings: TrackerSettings,
) -> PlateRead:
    groups, plate_bbox = _build_digit_groups(plate_model, crop, settings)
    if not groups:
        return PlateRead(None, 0.0, plate_bbox)
    best_group = max(groups, key=_group_score)
    plate_read = _group_to_plate_read(best_group, plate_bbox, settings.reads.min_digits)
    return plate_read if plate_read is not None else PlateRead(None, 0.0, plate_bbox)


def detect_all_plate_numbers(
    plate_model: YOLO | None,
    crop: np.ndarray,
    settings: TrackerSettings,
) -> list[PlateRead]:
    groups, plate_bbox = _build_digit_groups(plate_model, crop, settings)
    result = []
    for group in groups:
        plate_read = _group_to_plate_read(group, plate_bbox, settings.reads.min_digits)
        if plate_read is not None:
            result.append(plate_read)
    return result


def _draw_label(frame: np.ndarray, bbox: tuple[int, int, int, int], text: str) -> None:
    x1, y1, x2, _ = bbox
    cv2.rectangle(frame, (x1, y1), (x2, bbox[3]), (0, 200, 255), 2)
    if not text:
        return
    (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    top = max(0, y1 - height - baseline - 4)
    cv2.rectangle(frame, (x1, top), (x1 + width + 6, top + height + baseline + 4), (0, 200, 255), -1)
    cv2.putText(
        frame,
        text,
        (x1 + 3, top + height + 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )


def _save_plate_crop(
    artifacts: RunArtifacts,
    frame_index: int,
    tracker_id: int,
    crop: np.ndarray,
    plate_text: str | None = None,
    track_state: "TrackState | None" = None,
    plate_read: "PlateRead | None" = None,
) -> Path | None:
    if artifacts.debug_dir is None or crop.size == 0:
        return None
    debug_image = crop.copy()
    if plate_text:
        dest = artifacts.debug_dir / "resolved" / f"plate_{plate_text}"
    else:
        dest = artifacts.debug_dir / "unresolved"
    dest.mkdir(parents=True, exist_ok=True)
    stem = f"frame{frame_index}_tid{tracker_id}"
    output_path = dest / f"{stem}.jpg"
    cv2.imwrite(str(output_path), debug_image)

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
    (dest / f"{stem}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2)
    )
    return output_path


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


def _initialise_models(
    settings: TrackerSettings,
    base_dir: Path,
    collect_only: bool,
    logger: Logger,
) -> tuple[YOLO, YOLO | None, str]:
    device = get_device(settings.runtime.device)
    vehicle_model_path = resolve_runtime_path(base_dir, settings.models.vehicle_model)
    plate_model_path = resolve_runtime_path(base_dir, settings.models.plate_model)
    if vehicle_model_path is None or not vehicle_model_path.exists():
        raise FileNotFoundError(f"Vehicle model not found: {settings.models.vehicle_model}")
    vehicle_model = YOLO(str(vehicle_model_path))
    vehicle_model.to(device)
    plate_model = None
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
    return vehicle_model, plate_model, device


def _initialise_reid(settings: TrackerSettings, base_dir: Path, logger: Logger) -> ReIdentifier | None:
    if not settings.reid.enabled:
        return None
    from .reid import ReIdentifier

    gallery_path = resolve_runtime_path(base_dir, settings.reid.gallery_path)
    if gallery_path is None:
        return None
    logger(f"reid_gallery={gallery_path}")
    return ReIdentifier(str(gallery_path), device=settings.runtime.device, thresh=settings.reid.threshold)


def probe_source(source: str, settings: TrackerSettings) -> dict[str, object]:
    capture = _open_capture(source)
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or settings.runtime.source_fps_fallback
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return {"fps": fps, "width": width, "height": height}
    finally:
        capture.release()


def iter_source(
    source: str,
    mode: str,
    settings: TrackerSettings,
    source_stats: dict[str, object],
    stop_event: Event | None = None,
) -> Iterator[FramePacket]:
    stop_event = stop_event or Event()
    is_local_file = _is_local_file_source(source)
    frame_index = 0
    reconnects = 0
    fps = float(source_stats["fps"])
    width = int(source_stats["width"])
    height = int(source_stats["height"])
    stream_started_at = time.perf_counter()

    while not stop_event.is_set():
        capture = _open_capture(source)
        segment_started_at = time.perf_counter()
        segment_frame_index = 0
        while not stop_event.is_set():
            ok, frame = capture.read()
            if not ok:
                break
            frame_index += 1
            segment_frame_index += 1
            if mode == "stream":
                if is_local_file:
                    target_ts = segment_frame_index / max(fps, 1.0)
                    elapsed = time.perf_counter() - segment_started_at
                    if target_ts > elapsed:
                        time.sleep(target_ts - elapsed)
                timestamp = time.perf_counter() - stream_started_at
            else:
                ts_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
                timestamp = (ts_ms / 1000.0) if ts_ms else frame_index / max(fps, 1.0)
            yield FramePacket(
                frame=frame,
                frame_index=frame_index,
                timestamp=timestamp,
                fps=fps,
                width=width,
                height=height,
            )
        capture.release()
        if mode == "file":
            if settings.stream.loop_file:
                continue
            break
        reconnects += 1
        source_stats["reconnects"] = reconnects
        if settings.stream.max_reconnects >= 0 and reconnects > settings.stream.max_reconnects:
            break
        time.sleep(settings.stream.reconnect_delay_sec)


def process_source(
    source: str,
    mode: str,
    settings: TrackerSettings,
    base_dir: Path,
    output_dir: str | Path | None = None,
    collect_only: bool = False,
    stop_event: Event | None = None,
    limit_frames: int | None = None,
    calibrate_line: bool = False,
    logger: Logger | None = None,
) -> dict[str, object]:
    logger = logger or _default_logger
    stop_event = stop_event or Event()
    source_stats = probe_source(source, settings)
    source_stats["reconnects"] = 0
    packet_iter = iter_source(source, mode, settings, source_stats=source_stats, stop_event=stop_event)
    try:
        first_packet = next(packet_iter)
    except StopIteration as exc:
        raise RuntimeError(f"No frames received from source: {source}") from exc
    line_value = settings.line.value
    if calibrate_line:
        picked = pick_line_on_frame(first_packet.frame)
        if picked is None:
            raise RuntimeError("Line calibration cancelled")
        (x1, y1), (x2, y2) = picked
        line_value = to_percent_str(x1, y1, x2, y2, first_packet.width, first_packet.height)
        logger(f"calibrated_line={line_value}")
    lx1, ly1, lx2, ly2 = parse_line_arg(line_value, first_packet.width, first_packet.height)
    line_a = (lx1, ly1)
    line_b = (lx2, ly2)

    prefix = "collect" if collect_only else f"detect_{mode}"
    artifacts = prepare_artifacts(
        output_dir=output_dir,
        prefix=prefix,
        write_video=settings.output.write_video,
        write_csv=settings.output.write_csv,
        write_jsonl=settings.output.write_jsonl,
        write_summary=settings.output.write_summary,
        save_plate_crops=settings.output.save_plate_crops and not collect_only,
    )

    if collect_only:
        csv_fields = [
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
    else:
        csv_fields = [
            "timestamp",
            "frame_index",
            "tracker_id",
            "rider_id",
            "identity_source",
            "plate_text",
            "plate_conf",
            "lap",
            "lap_time",
            "bbox_x1",
            "bbox_y1",
            "bbox_x2",
            "bbox_y2",
            "center_x",
            "center_y",
            "crop_file",
        ]
    csv_writer = CsvWriter(artifacts.csv_path, csv_fields)
    jsonl_writer = JsonlWriter(artifacts.jsonl_path)

    vehicle_model, plate_model, device = _initialise_models(settings, base_dir, collect_only, logger)
    tracker_yaml = resolve_tracker_path(settings.models.tracker, base_dir)
    reid = None if collect_only else _initialise_reid(settings, base_dir, logger)

    writer = None
    if artifacts.video_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(artifacts.video_path),
            fourcc,
            max(first_packet.fps, 1.0),
            (first_packet.width, first_packet.height),
        )

    track_states: dict[int, TrackState] = {}
    crossing_counts: dict[str, int] = {}
    unresolved_crossings = 0
    saved_crops = 0
    events = 0
    status_started_at = time.perf_counter()

    for packet in chain((first_packet,), packet_iter):
        if stop_event.is_set():
            break
        if limit_frames is not None and packet.frame_index > limit_frames:
            break
        stale_ids = [
            tracker_id
            for tracker_id, state in track_states.items()
            if (packet.timestamp - state.last_seen_ts) > settings.reads.track_state_ttl_sec
        ]
        for tracker_id in stale_ids:
            del track_states[tracker_id]

        frame = packet.frame.copy()
        result = vehicle_model.track(
            frame,
            tracker=tracker_yaml,
            persist=True,
            device=device,
            conf=settings.models.vehicle_conf,
            iou=settings.models.vehicle_iou,
            imgsz=settings.models.vehicle_imgsz,
            verbose=False,
        )[0]

        frame_records: list[DetectionRecord] = []
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0 and boxes.id is not None:
            xyxy = _xyxy_array(boxes, "xyxy")
            classes = _xyxy_array(boxes, "cls").astype(int)
            tracker_ids = _xyxy_array(boxes, "id").astype(int)
            for bbox_raw, class_id, tracker_id in zip(xyxy, classes, tracker_ids):
                if class_id != settings.models.motorcycle_class_id:
                    continue
                x1, y1, x2, y2 = map(int, bbox_raw.tolist())
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                track_state = track_states.setdefault(int(tracker_id), TrackState())
                track_state.last_seen_ts = packet.timestamp

                _, distance_to_line = point_line_side_and_dist(center, line_a, line_b)
                read = PlateRead(None, 0.0, None)
                should_scan = (
                    packet.frame_index % max(settings.reads.scan_every_n_frames, 1) == 0
                    and distance_to_line <= (settings.line.width * settings.line.read_distance_multiplier)
                )
                bike_crop = None
                if should_scan:
                    bx1, by1, bx2, by2 = expand_bbox(
                        x1,
                        y1,
                        x2,
                        y2,
                        packet.width,
                        packet.height,
                        1.0 + settings.models.bike_crop_expand,
                    )
                    bike_crop = packet.frame[by1:by2, bx1:bx2]
                    if not collect_only:
                        plate_input = zone_crop(
                            bike_crop,
                            settings.models.plate_zone_n,
                            settings.models.plate_zone_select,
                        )
                        if min(plate_input.shape[:2]) >= settings.models.min_bike_crop_px:
                            read = detect_plate_number(plate_model, plate_input, settings)
                        if read.text is not None:
                            track_state.add_observation(
                                PlateObservation(
                                    text=read.text,
                                    confidence=read.confidence,
                                    timestamp=packet.timestamp,
                                    frame_index=packet.frame_index,
                                ),
                                ttl_sec=settings.reads.track_state_ttl_sec,
                            )
                frame_records.append(
                    DetectionRecord(
                        tracker_id=int(tracker_id),
                        bbox=(x1, y1, x2, y2),
                        center=center,
                        plate_text=read.text,
                        plate_conf=read.confidence,
                    )
                )

                crossing: CrossingDecision = detect_crossing(
                    track_state.last_center,
                    center,
                    line_a,
                    line_b,
                    settings.line.width,
                    settings.line.direction,
                )
                track_state.last_center = center
                track_state.prune(packet.timestamp, settings.reads.track_state_ttl_sec)
                if not crossing.crossed:
                    continue
                if (packet.timestamp - track_state.last_cross_ts) < settings.line.cooldown_sec:
                    continue
                track_state.last_cross_ts = packet.timestamp

                bx1, by1, bx2, by2 = expand_bbox(
                    x1,
                    y1,
                    x2,
                    y2,
                    packet.width,
                    packet.height,
                    1.0 + settings.models.bike_crop_expand,
                )
                bike_crop = packet.frame[by1:by2, bx1:bx2]

                if collect_only:
                    crop_path = artifacts.run_dir / f"frame{packet.frame_index}_tid{tracker_id}.jpg"
                    cv2.imwrite(str(crop_path), bike_crop)
                    metadata = {
                        "timestamp": round(packet.timestamp, 3),
                        "frame_index": packet.frame_index,
                        "tracker_id": int(tracker_id),
                        "bbox": [x1, y1, x2, y2],
                        "center": [center[0], center[1]],
                        "crop_path": str(crop_path),
                    }
                    jsonl_writer.write(metadata)
                    csv_writer.write(
                        _flatten_collect_row(
                            packet.timestamp,
                            packet.frame_index,
                            int(tracker_id),
                            (x1, y1, x2, y2),
                            center,
                            crop_path,
                        )
                    )
                    saved_crops += 1
                    continue

                plate_debug = zone_crop(
                    bike_crop,
                    settings.models.plate_zone_n,
                    settings.models.plate_zone_select,
                )

                plate_groups = detect_all_plate_numbers(plate_model, plate_debug, settings)
                if not plate_groups:
                    chosen = track_state.best_plate(packet.timestamp, settings.reads.vote_window_sec)
                    if chosen is not None:
                        plate_groups = [PlateRead(chosen.text, chosen.confidence, None)]

                if len(plate_groups) > 1:
                    reverse = settings.line.direction == "left_to_right"
                    plate_groups.sort(key=lambda pr: pr.x_center, reverse=reverse)

                saved_path = _save_plate_crop(
                    artifacts,
                    packet.frame_index,
                    int(tracker_id),
                    plate_debug,
                    plate_text=plate_groups[0].text if plate_groups else None,
                    track_state=track_state,
                    plate_read=read,
                )
                if saved_path is not None:
                    saved_crops += 1

                crop_file = str(saved_path) if saved_path else ""

                if not plate_groups:
                    reid_id = reid.identify(bike_crop) if reid is not None else None
                    if reid_id is None:
                        unresolved_crossings += 1
                        unresolved_event: dict[str, object] = {
                            "timestamp": round(packet.timestamp, 3),
                            "frame_index": packet.frame_index,
                            "tracker_id": int(tracker_id),
                            "rider_id": "",
                            "identity_source": "unresolved",
                            "plate_text": "",
                            "plate_conf": 0.0,
                            "lap": "",
                            "lap_time": "",
                            "bbox": [x1, y1, x2, y2],
                            "center": [center[0], center[1]],
                            "crop_file": crop_file,
                        }
                        jsonl_writer.write(unresolved_event)
                        csv_writer.write(_flatten_event(dict(unresolved_event)))
                    else:
                        crossing_counts[reid_id] = crossing_counts.get(reid_id, 0) + 1
                        reid_event: dict[str, object] = {
                            "timestamp": round(packet.timestamp, 3),
                            "frame_index": packet.frame_index,
                            "tracker_id": int(tracker_id),
                            "rider_id": reid_id,
                            "identity_source": "reid",
                            "plate_text": "",
                            "plate_conf": 0.0,
                            "lap": "",
                            "lap_time": "",
                            "bbox": [x1, y1, x2, y2],
                            "center": [center[0], center[1]],
                            "crop_file": crop_file,
                        }
                        jsonl_writer.write(reid_event)
                        csv_writer.write(_flatten_event(dict(reid_event)))
                        events += 1
                else:
                    for group_idx, plate_read in enumerate(plate_groups):
                        ts = packet.timestamp + group_idx * 0.1
                        rider_id = f"plate_{plate_read.text}"
                        crossing_counts[rider_id] = crossing_counts.get(rider_id, 0) + 1
                        event: dict[str, object] = {
                            "timestamp": round(ts, 3),
                            "frame_index": packet.frame_index,
                            "tracker_id": int(tracker_id),
                            "rider_id": rider_id,
                            "identity_source": "plate",
                            "plate_text": plate_read.text,
                            "plate_conf": round(plate_read.confidence, 3),
                            "lap": "",
                            "lap_time": "",
                            "bbox": [x1, y1, x2, y2],
                            "center": [center[0], center[1]],
                            "crop_file": crop_file,
                        }
                        jsonl_writer.write(event)
                        csv_writer.write(_flatten_event(dict(event)))
                        events += 1

        cv2.line(frame, line_a, line_b, (255, 255, 255), settings.line.width)
        for record in frame_records:
            label = f"tid={record.tracker_id}"
            if record.plate_text:
                label += f" plate={record.plate_text}:{record.plate_conf:.2f}"
            _draw_label(frame, record.bbox, label)
        if not collect_only:
            top_rows = sorted(crossing_counts.items(), key=lambda item: (-item[1], item[0]))[: settings.output.overlay_top_n]
            for index, (rider_id, count) in enumerate(top_rows):
                cv2.putText(
                    frame,
                    f"{rider_id} x{count}",
                    (10, 28 + index * 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
        if writer is not None:
            writer.write(frame)
        if (time.perf_counter() - status_started_at) >= settings.stream.status_interval_sec:
            logger(
                f"frame={packet.frame_index} ts={packet.timestamp:.2f}s "
                f"events={events} unresolved={unresolved_crossings} saved_crops={saved_crops}"
            )
            status_started_at = time.perf_counter()

    if writer is not None:
        writer.release()
    csv_writer.close()
    jsonl_writer.close()

    if artifacts.summary_path is not None:
        summary = {
            "mode": mode,
            "source": source,
            "line": line_value,
            "run_dir": str(artifacts.run_dir),
            "video_path": str(artifacts.video_path) if artifacts.video_path is not None else None,
            "csv_path": str(artifacts.csv_path) if artifacts.csv_path is not None else None,
            "jsonl_path": str(artifacts.jsonl_path) if artifacts.jsonl_path is not None else None,
            "events": events,
            "saved_crops": saved_crops,
            "unresolved_crossings": unresolved_crossings,
            "crossing_counts": crossing_counts,
            "source_stats": source_stats,
        }
        artifacts.summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    return {
        "mode": mode,
        "source": source,
        "line": line_value,
        "run_dir": str(artifacts.run_dir),
        "video_path": str(artifacts.video_path) if artifacts.video_path is not None else None,
        "csv_path": str(artifacts.csv_path) if artifacts.csv_path is not None else None,
        "jsonl_path": str(artifacts.jsonl_path) if artifacts.jsonl_path is not None else None,
        "events": events,
        "saved_crops": saved_crops,
        "unresolved_crossings": unresolved_crossings,
        "crossing_counts": crossing_counts,
        "source_stats": source_stats,
    }


def run_file_detection(
    source: str,
    settings: TrackerSettings,
    base_dir: Path,
    output_dir: str | Path | None = None,
    limit_frames: int | None = None,
    calibrate_line: bool = False,
    logger: Logger | None = None,
) -> dict[str, object]:
    return process_source(
        source=source,
        mode="file",
        settings=settings,
        base_dir=base_dir,
        output_dir=output_dir,
        limit_frames=limit_frames,
        calibrate_line=calibrate_line,
        logger=logger,
    )


def run_stream_detection(
    source: str,
    settings: TrackerSettings,
    base_dir: Path,
    output_dir: str | Path | None = None,
    stop_event: Event | None = None,
    limit_frames: int | None = None,
    calibrate_line: bool = False,
    logger: Logger | None = None,
) -> dict[str, object]:
    return process_source(
        source=source,
        mode="stream",
        settings=settings,
        base_dir=base_dir,
        output_dir=output_dir,
        stop_event=stop_event,
        limit_frames=limit_frames,
        calibrate_line=calibrate_line,
        logger=logger,
    )


def collect_samples(
    source: str,
    mode: str,
    settings: TrackerSettings,
    base_dir: Path,
    output_dir: str | Path | None = None,
    stop_event: Event | None = None,
    limit_frames: int | None = None,
    calibrate_line: bool = False,
    logger: Logger | None = None,
) -> dict[str, object]:
    return process_source(
        source=source,
        mode=mode,
        settings=settings,
        base_dir=base_dir,
        output_dir=output_dir,
        collect_only=True,
        stop_event=stop_event,
        limit_frames=limit_frames,
        calibrate_line=calibrate_line,
        logger=logger,
    )
