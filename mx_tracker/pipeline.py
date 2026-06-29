"""Detection pipeline orchestration.

`SourceProcessor` ties the layers together: it pulls frames from a
`VideoSource`, runs the `DetectionModels`, decides crossings against the finish
line, and routes results to the `EventLog`, `CropArchive` and `OverlayWriter`.

The public entry points (`run_file_detection`, `run_stream_detection`,
`collect_samples`) are thin wrappers around it. Domain types and pure helpers
are re-exported here for backwards compatibility with existing imports.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from itertools import chain
from pathlib import Path
from threading import Event
from typing import Callable

import cv2

from .artifacts import (
    CropArchive,
    EventLog,
    RunArtifacts,
    _flatten_collect_row,
    _flatten_event,
    prepare_artifacts,
)
from .config import TrackerSettings
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
from .models import DetectionModels
from .overlay import OverlayWriter
from .plate_reading import (
    PlateRead,
    _build_digit_groups,
    _group_score,
    _group_to_plate_read,
    _xyxy_array,
    detect_all_plate_numbers,
    detect_plate_number,
)
from .tracking import DetectionRecord, PlateObservation, TrackState
from .video_source import FramePacket, VideoSource

Logger = Callable[[str], None]

# Re-exported so `from mx_tracker.pipeline import ...` keeps working for tests
# and any external callers after the module split.
__all__ = [
    "PlateObservation",
    "PlateRead",
    "TrackState",
    "DetectionRecord",
    "FramePacket",
    "RunArtifacts",
    "StageProfiler",
    "SourceProcessor",
    "detect_plate_number",
    "detect_all_plate_numbers",
    "collect_samples",
    "run_file_detection",
    "run_stream_detection",
    "process_source",
    "_flatten_event",
    "_flatten_collect_row",
    "_group_score",
    "_group_to_plate_read",
    "_build_digit_groups",
]


def _default_logger(message: str) -> None:
    print(message)


class StageProfiler:
    """Accumulates per-stage timings and emits periodic ms/frame summaries."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.reset()

    def reset(self) -> None:
        self.frames = 0
        self.totals: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)
        self.started_at = time.perf_counter()

    def frame(self) -> None:
        if self.enabled:
            self.frames += 1

    def add(self, name: str, elapsed: float) -> None:
        if not self.enabled:
            return
        self.totals[name] += elapsed
        self.counts[name] += 1

    def summary_lines(self) -> list[str]:
        if not self.enabled or self.frames <= 0:
            return []
        elapsed = max(time.perf_counter() - self.started_at, 1e-9)

        def ms_per_frame(name: str) -> float:
            return self.totals.get(name, 0.0) * 1000.0 / self.frames

        def calls(name: str) -> int:
            return self.counts.get(name, 0)

        lines = [
            "profile ms/frame "
            f"wall={elapsed * 1000.0 / self.frames:.1f} "
            f"total={ms_per_frame('frame_total'):.1f} "
            f"copy={ms_per_frame('frame_copy'):.1f} "
            f"vehicle={ms_per_frame('vehicle_track'):.1f} "
            f"logic={ms_per_frame('track_logic'):.1f} "
            f"scan={ms_per_frame('plate_scan'):.1f} "
            f"cross={ms_per_frame('crossing_read'):.1f} "
            f"reid={ms_per_frame('reid'):.1f} "
            f"save={ms_per_frame('save_crop'):.1f} "
            f"io={ms_per_frame('event_io'):.1f} "
            f"overlay={ms_per_frame('overlay_write'):.1f}",
            "profile calls "
            f"frames={self.frames} "
            f"scan={calls('plate_scan')} "
            f"cross={calls('crossing_read')} "
            f"reid={calls('reid')} "
            f"save={calls('save_crop')} "
            f"io={calls('event_io')} "
            f"overlay={calls('overlay_write')}",
        ]
        self.reset()
        return lines


class _Timer:
    """Context manager that records its elapsed time into a StageProfiler stage."""

    __slots__ = ("_profiler", "_name", "_start")

    def __init__(self, profiler: StageProfiler, name: str) -> None:
        self._profiler = profiler
        self._name = name

    def __enter__(self) -> "_Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self._profiler.add(self._name, time.perf_counter() - self._start)


class SourceProcessor:
    """Runs detection (or crop collection) over a single video source.

    One instance corresponds to one run: it owns the per-run state (track
    states, crossing counts, output writers) and produces a summary dict.
    """

    def __init__(
        self,
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
    ) -> None:
        self.source = source
        self.mode = mode
        self.settings = settings
        self.base_dir = base_dir
        self.output_dir = output_dir
        self.collect_only = collect_only
        self.stop_event = stop_event or Event()
        self.limit_frames = limit_frames
        self.calibrate_line = calibrate_line
        self.log = logger or _default_logger

        # Per-run state, populated in run().
        self.track_states: dict[int, TrackState] = {}
        self.crossing_counts: dict[str, int] = {}
        self.recent_crossings: dict[str, tuple[str, float]] = {}  # rider_id → (display_text, ts)
        self.unresolved_crossings = 0
        self.saved_crops = 0
        self.events = 0

    # -- run lifecycle -----------------------------------------------------

    def run(self) -> dict[str, object]:
        video = VideoSource(self.source, self.mode, self.settings)
        source_stats = video.probe()
        packet_iter = video.frames(self.stop_event)
        try:
            first_packet = next(packet_iter)
        except StopIteration as exc:
            raise RuntimeError(f"No frames received from source: {self.source}") from exc

        line_a, line_b, line_value = self._resolve_line(first_packet)
        artifacts = self._prepare_artifacts()

        wall_started_at = datetime.now(timezone.utc)
        self._write_run_info(artifacts, wall_started_at)
        event_log = EventLog(artifacts, wall_started_at, self.collect_only)
        crop_archive = CropArchive(artifacts.debug_dir)
        models = DetectionModels.load(self.settings, self.base_dir, self.collect_only, self.log)
        overlay = OverlayWriter(
            artifacts.video_path,
            self.settings,
            line_a,
            line_b,
            first_packet.fps,
            first_packet.width,
            first_packet.height,
            self.collect_only,
        )
        profiler = StageProfiler(self.settings.runtime.profile)
        status_started_at = time.perf_counter()

        for packet in chain((first_packet,), packet_iter):
            frame_started_at = time.perf_counter()
            profiler.frame()
            if self.stop_event.is_set():
                break
            if self.limit_frames is not None and packet.frame_index > self.limit_frames:
                break
            self._evict_stale_tracks(packet.timestamp)
            self._process_frame(packet, line_a, line_b, models, event_log, crop_archive, overlay, profiler, artifacts)
            profiler.add("frame_total", time.perf_counter() - frame_started_at)
            if (time.perf_counter() - status_started_at) >= self.settings.stream.status_interval_sec:
                self._log_status(packet)
                for line in profiler.summary_lines():
                    self.log(line)
                status_started_at = time.perf_counter()

        overlay.close()
        event_log.close()
        return self._finalize(artifacts, line_value, source_stats)

    # -- setup helpers -----------------------------------------------------

    def _resolve_line(self, first_packet: FramePacket) -> tuple[tuple[int, int], tuple[int, int], str]:
        line_value = self.settings.line.value
        if self.calibrate_line:
            picked = pick_line_on_frame(first_packet.frame)
            if picked is None:
                raise RuntimeError("Line calibration cancelled")
            (x1, y1), (x2, y2) = picked
            line_value = to_percent_str(x1, y1, x2, y2, first_packet.width, first_packet.height)
            self.log(f"calibrated_line={line_value}")
        lx1, ly1, lx2, ly2 = parse_line_arg(line_value, first_packet.width, first_packet.height)
        return (lx1, ly1), (lx2, ly2), line_value

    def _prepare_artifacts(self) -> RunArtifacts:
        prefix = "collect" if self.collect_only else f"detect_{self.mode}"
        return prepare_artifacts(
            output_dir=self.output_dir,
            prefix=prefix,
            write_video=self.settings.output.write_video,
            write_csv=self.settings.output.write_csv,
            write_jsonl=self.settings.output.write_jsonl,
            write_summary=self.settings.output.write_summary,
            save_plate_crops=self.settings.output.save_plate_crops and not self.collect_only,
        )

    def _write_run_info(self, artifacts: RunArtifacts, wall_started_at: datetime) -> None:
        (artifacts.run_dir / "run_info.json").write_text(
            json.dumps(
                {"started_at": wall_started_at.isoformat(timespec="seconds"), "source": self.source},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    # -- per-frame processing ---------------------------------------------

    def _evict_stale_tracks(self, now_ts: float) -> None:
        ttl = self.settings.reads.track_state_ttl_sec
        stale_ids = [tid for tid, state in self.track_states.items() if (now_ts - state.last_seen_ts) > ttl]
        for tracker_id in stale_ids:
            del self.track_states[tracker_id]

    def _process_frame(
        self,
        packet: FramePacket,
        line_a: tuple[int, int],
        line_b: tuple[int, int],
        models: DetectionModels,
        event_log: EventLog,
        crop_archive: CropArchive,
        overlay: OverlayWriter,
        profiler: StageProfiler,
        artifacts: RunArtifacts,
    ) -> None:
        with _Timer(profiler, "frame_copy"):
            frame = packet.frame.copy()
        with _Timer(profiler, "vehicle_track"):
            result = models.track(frame, self.settings)

        logic_start = time.perf_counter()
        frame_records: list[DetectionRecord] = []
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0 and boxes.id is not None:
            xyxy = _xyxy_array(boxes, "xyxy")
            classes = _xyxy_array(boxes, "cls").astype(int)
            tracker_ids = _xyxy_array(boxes, "id").astype(int)
            for bbox_raw, class_id, tracker_id in zip(xyxy, classes, tracker_ids):
                if class_id != self.settings.models.motorcycle_class_id:
                    continue
                self._process_track(
                    packet,
                    int(tracker_id),
                    tuple(map(int, bbox_raw.tolist())),
                    line_a,
                    line_b,
                    models,
                    event_log,
                    crop_archive,
                    profiler,
                    artifacts,
                    frame_records,
                )
        profiler.add("track_logic", time.perf_counter() - logic_start)

        with _Timer(profiler, "overlay_write"):
            overlay.render(frame, packet, frame_records, self.crossing_counts, self.recent_crossings)

    def _process_track(
        self,
        packet: FramePacket,
        tracker_id: int,
        bbox: tuple[int, int, int, int],
        line_a: tuple[int, int],
        line_b: tuple[int, int],
        models: DetectionModels,
        event_log: EventLog,
        crop_archive: CropArchive,
        profiler: StageProfiler,
        artifacts: RunArtifacts,
        frame_records: list[DetectionRecord],
    ) -> None:
        x1, y1, x2, y2 = bbox
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        track_state = self.track_states.setdefault(tracker_id, TrackState())
        track_state.last_seen_ts = packet.timestamp

        _, distance_to_line = point_line_side_and_dist(center, line_a, line_b)
        scan_read = self._scan_plate(packet, bbox, distance_to_line, track_state, models, profiler)
        frame_records.append(
            DetectionRecord(
                tracker_id=tracker_id,
                bbox=bbox,
                center=center,
                plate_text=scan_read.text,
                plate_conf=scan_read.confidence,
            )
        )

        crossing: CrossingDecision = detect_crossing(
            track_state.last_center,
            center,
            line_a,
            line_b,
            self.settings.line.width,
            self.settings.line.direction,
        )
        track_state.last_center = center
        track_state.prune(packet.timestamp, self.settings.reads.track_state_ttl_sec)
        if not crossing.crossed:
            return
        if (packet.timestamp - track_state.last_cross_ts) < self.settings.line.cooldown_sec:
            return
        track_state.last_cross_ts = packet.timestamp

        self._handle_crossing(
            packet, tracker_id, bbox, center, track_state, scan_read, models, event_log, crop_archive, profiler, artifacts
        )

    def _scan_plate(
        self,
        packet: FramePacket,
        bbox: tuple[int, int, int, int],
        distance_to_line: float,
        track_state: TrackState,
        models: DetectionModels,
        profiler: StageProfiler,
    ) -> PlateRead:
        """Opportunistic per-frame plate read while a track is near the line."""
        read = PlateRead(None, 0.0, None)
        should_scan = (
            packet.frame_index % max(self.settings.reads.scan_every_n_frames, 1) == 0
            and distance_to_line <= (self.settings.line.width * self.settings.line.read_distance_multiplier)
        )
        if not should_scan:
            return read

        bike_crop = self._bike_crop(packet, bbox)
        if self.collect_only:
            return read

        plate_input = zone_crop(bike_crop, self.settings.models.plate_zone_n, self.settings.models.plate_zone_select)
        if min(plate_input.shape[:2]) >= self.settings.models.min_bike_crop_px:
            with _Timer(profiler, "plate_scan"):
                read = models.plate_reader.read_best(plate_input)
        if read.text is not None:
            track_state.add_observation(
                PlateObservation(
                    text=read.text,
                    confidence=read.confidence,
                    timestamp=packet.timestamp,
                    frame_index=packet.frame_index,
                ),
                ttl_sec=self.settings.reads.track_state_ttl_sec,
            )
        return read

    def _bike_crop(self, packet: FramePacket, bbox: tuple[int, int, int, int]):
        bx1, by1, bx2, by2 = expand_bbox(
            bbox[0], bbox[1], bbox[2], bbox[3], packet.width, packet.height, 1.0 + self.settings.models.bike_crop_expand
        )
        return packet.frame[by1:by2, bx1:bx2]

    def _handle_crossing(
        self,
        packet: FramePacket,
        tracker_id: int,
        bbox: tuple[int, int, int, int],
        center: tuple[int, int],
        track_state: TrackState,
        scan_read: PlateRead,
        models: DetectionModels,
        event_log: EventLog,
        crop_archive: CropArchive,
        profiler: StageProfiler,
        artifacts: RunArtifacts,
    ) -> None:
        bike_crop = self._bike_crop(packet, bbox)

        if self.collect_only:
            crop_path = artifacts.run_dir / f"frame{packet.frame_index}_tid{tracker_id}.jpg"
            cv2.imwrite(str(crop_path), bike_crop)
            event_log.emit_collect(
                timestamp=packet.timestamp,
                frame_index=packet.frame_index,
                tracker_id=tracker_id,
                bbox=bbox,
                center=center,
                crop_path=crop_path,
            )
            self.saved_crops += 1
            return

        plate_debug = zone_crop(bike_crop, self.settings.models.plate_zone_n, self.settings.models.plate_zone_select)
        with _Timer(profiler, "crossing_read"):
            plate_groups = models.plate_reader.read_all(plate_debug)
        if not plate_groups:
            chosen = track_state.best_plate(packet.timestamp, self.settings.reads.vote_window_sec)
            if chosen is not None:
                plate_groups = [PlateRead(chosen.text, chosen.confidence, None)]
        if len(plate_groups) > 1:
            reverse = self.settings.line.direction == "left_to_right"
            plate_groups.sort(key=lambda pr: pr.x_center, reverse=reverse)

        if not plate_groups:
            self._emit_unresolved_or_reid(
                packet, tracker_id, bbox, center, track_state, scan_read, bike_crop, plate_debug,
                models, event_log, crop_archive, profiler,
            )
        else:
            self._emit_plate_groups(
                packet, tracker_id, bbox, center, track_state, scan_read, plate_groups, plate_debug,
                event_log, crop_archive, profiler,
            )

    def _emit_unresolved_or_reid(
        self,
        packet: FramePacket,
        tracker_id: int,
        bbox: tuple[int, int, int, int],
        center: tuple[int, int],
        track_state: TrackState,
        scan_read: PlateRead,
        bike_crop,
        plate_debug,
        models: DetectionModels,
        event_log: EventLog,
        crop_archive: CropArchive,
        profiler: StageProfiler,
    ) -> None:
        with _Timer(profiler, "save_crop"):
            saved_path = crop_archive.save(
                packet.frame_index,
                tracker_id,
                plate_debug,
                plate_text=None,
                track_state=track_state,
                plate_read=scan_read,
                bike_crop=bike_crop if self.settings.output.save_bike_crops_unresolved else None,
            )
        if saved_path is not None:
            self.saved_crops += 1
        crop_file = str(saved_path) if saved_path else ""

        with _Timer(profiler, "reid"):
            reid_id = models.reid.identify(bike_crop) if models.reid is not None else None

        if reid_id is None:
            self.unresolved_crossings += 1
            with _Timer(profiler, "event_io"):
                event_log.emit_crossing(
                    timestamp=packet.timestamp,
                    frame_index=packet.frame_index,
                    tracker_id=tracker_id,
                    rider_id="",
                    identity_source="unresolved",
                    plate_text="",
                    plate_conf=0.0,
                    bbox=bbox,
                    center=center,
                    crop_file=crop_file,
                )
        else:
            self.crossing_counts[reid_id] = self.crossing_counts.get(reid_id, 0) + 1
            self.recent_crossings[reid_id] = (reid_id.removeprefix("plate_"), packet.timestamp)
            with _Timer(profiler, "event_io"):
                event_log.emit_crossing(
                    timestamp=packet.timestamp,
                    frame_index=packet.frame_index,
                    tracker_id=tracker_id,
                    rider_id=reid_id,
                    identity_source="reid",
                    plate_text="",
                    plate_conf=0.0,
                    bbox=bbox,
                    center=center,
                    crop_file=crop_file,
                )
            self.events += 1

    def _emit_plate_groups(
        self,
        packet: FramePacket,
        tracker_id: int,
        bbox: tuple[int, int, int, int],
        center: tuple[int, int],
        track_state: TrackState,
        scan_read: PlateRead,
        plate_groups: list[PlateRead],
        plate_debug,
        event_log: EventLog,
        crop_archive: CropArchive,
        profiler: StageProfiler,
    ) -> None:
        for group_idx, plate_read in enumerate(plate_groups):
            with _Timer(profiler, "save_crop"):
                saved_path = crop_archive.save(
                    packet.frame_index,
                    tracker_id,
                    plate_debug,
                    plate_text=plate_read.text,
                    track_state=track_state,
                    plate_read=scan_read,
                )
            if saved_path is not None:
                self.saved_crops += 1
            crop_file = str(saved_path) if saved_path else ""
            ts = packet.timestamp + group_idx * 0.1
            rider_id = f"plate_{plate_read.text}"
            self.crossing_counts[rider_id] = self.crossing_counts.get(rider_id, 0) + 1
            self.recent_crossings[rider_id] = (plate_read.text, packet.timestamp)
            with _Timer(profiler, "event_io"):
                event_log.emit_crossing(
                    timestamp=ts,
                    frame_index=packet.frame_index,
                    tracker_id=tracker_id,
                    rider_id=rider_id,
                    identity_source="plate",
                    plate_text=plate_read.text,
                    plate_conf=plate_read.confidence,
                    bbox=bbox,
                    center=center,
                    crop_file=crop_file,
                )
            self.events += 1

    # -- reporting ---------------------------------------------------------

    def _log_status(self, packet: FramePacket) -> None:
        self.log(
            f"frame={packet.frame_index} ts={packet.timestamp:.2f}s "
            f"events={self.events} unresolved={self.unresolved_crossings} saved_crops={self.saved_crops}"
        )

    def _finalize(self, artifacts: RunArtifacts, line_value: str, source_stats: dict[str, object]) -> dict[str, object]:
        summary = {
            "mode": self.mode,
            "source": self.source,
            "line": line_value,
            "run_dir": str(artifacts.run_dir),
            "video_path": str(artifacts.video_path) if artifacts.video_path is not None else None,
            "csv_path": str(artifacts.csv_path) if artifacts.csv_path is not None else None,
            "jsonl_path": str(artifacts.jsonl_path) if artifacts.jsonl_path is not None else None,
            "events": self.events,
            "saved_crops": self.saved_crops,
            "unresolved_crossings": self.unresolved_crossings,
            "crossing_counts": self.crossing_counts,
            "source_stats": source_stats,
        }
        if artifacts.summary_path is not None:
            artifacts.summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        return summary


# -- public entry points ---------------------------------------------------


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
    return SourceProcessor(
        source=source,
        mode=mode,
        settings=settings,
        base_dir=base_dir,
        output_dir=output_dir,
        collect_only=collect_only,
        stop_event=stop_event,
        limit_frames=limit_frames,
        calibrate_line=calibrate_line,
        logger=logger,
    ).run()


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
