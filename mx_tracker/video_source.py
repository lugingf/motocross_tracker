"""Frame acquisition from files, streams and cameras.

`VideoSource` hides the difference between a finite file and a live/looping
stream behind a single `frames()` iterator. It also owns reconnect handling
and timestamp derivation so the pipeline only ever sees `FramePacket`s.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Iterator

import cv2
import numpy as np

from .config import TrackerSettings


@dataclass(slots=True)
class FramePacket:
    frame: np.ndarray
    frame_index: int
    timestamp: float
    fps: float
    width: int
    height: int


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


class VideoSource:
    """A reconnecting, mode-aware source of video frames.

    `mode` is ``"file"`` (finite, optionally looping) or ``"stream"`` (live or
    simulated, with reconnects). Call `probe()` once to read dimensions/fps,
    then iterate `frames()`.
    """

    def __init__(self, source: str, mode: str, settings: TrackerSettings) -> None:
        self.source = source
        self.mode = mode
        self.settings = settings
        self.stats: dict[str, object] = {}

    def probe(self) -> dict[str, object]:
        capture = _open_capture(self.source)
        try:
            fps = capture.get(cv2.CAP_PROP_FPS) or self.settings.runtime.source_fps_fallback
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            capture.release()
        self.stats = {"fps": fps, "width": width, "height": height, "reconnects": 0}
        return self.stats

    def frames(self, stop_event: Event | None = None) -> Iterator[FramePacket]:
        if not self.stats:
            self.probe()
        stop_event = stop_event or Event()
        is_local_file = _is_local_file_source(self.source)
        frame_index = 0
        reconnects = 0
        fps = float(self.stats["fps"])
        width = int(self.stats["width"])
        height = int(self.stats["height"])
        stream_started_at = time.perf_counter()

        while not stop_event.is_set():
            capture = _open_capture(self.source)
            segment_started_at = time.perf_counter()
            segment_frame_index = 0
            while not stop_event.is_set():
                ok, frame = capture.read()
                if not ok:
                    break
                frame_index += 1
                segment_frame_index += 1
                if self.mode == "stream":
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
            if self.mode == "file":
                if self.settings.stream.loop_file:
                    continue
                break
            reconnects += 1
            self.stats["reconnects"] = reconnects
            if self.settings.stream.max_reconnects >= 0 and reconnects > self.settings.stream.max_reconnects:
                break
            time.sleep(self.settings.stream.reconnect_delay_sec)
