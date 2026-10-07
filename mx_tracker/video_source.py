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
from urllib.parse import urlsplit, urlunsplit

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
    # Which unbroken piece of the stream the frame belongs to, and how much of
    # the timeline lies unseen between the previous piece and this frame.
    segment_id: int = 0
    gap_before: float = 0.0


_ROTATIONS = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}

NETWORK_SCHEMES = ("rtsp", "rtsps", "rtmp", "rtmps", "http", "https", "srt", "udp", "tcp")


def redact_source(value: str) -> str:
    """The address of a source without the credentials or the query that may carry them."""
    parts = urlsplit(value)
    if not parts.scheme or not parts.netloc:
        return value
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, host, parts.path, "", ""))


def _is_network_source(value: str) -> bool:
    return urlsplit(value).scheme.lower() in NETWORK_SCHEMES


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

    def _uses_pyav(self) -> bool:
        return (
            self.mode == "stream"
            and self.settings.stream.ingest == "pyav"
            and _is_network_source(self.source)
        )

    def _pyav_options(self) -> dict[str, str]:
        if urlsplit(self.source).scheme.lower() in ("rtsp", "rtsps"):
            return {"rtsp_transport": self.settings.stream.rtsp_transport}
        return {}

    def _open_pyav(self):
        import av

        stream = self.settings.stream
        container = av.open(
            self.source,
            options=self._pyav_options(),
            timeout=(stream.open_timeout_sec, stream.read_timeout_sec),
        )
        container.streams.video[0].thread_type = "AUTO"
        return container

    def _probe_pyav(self) -> dict[str, object]:
        try:
            container = self._open_pyav()
        except Exception as exc:
            raise RuntimeError(f"Cannot open source: {redact_source(self.source)}") from exc
        try:
            video = container.streams.video[0]
            rate = video.average_rate or video.guessed_rate
            fps = float(rate) if rate else self.settings.runtime.source_fps_fallback
            width = int(video.codec_context.width)
            height = int(video.codec_context.height)
        finally:
            container.close()
        self.stats = {"fps": fps, "width": width, "height": height, "reconnects": 0, "segments": 1}
        return self.stats

    def probe(self) -> dict[str, object]:
        if self._uses_pyav():
            return self._probe_pyav()
        capture = _open_capture(self.source)
        try:
            reported_fps = capture.get(cv2.CAP_PROP_FPS)
            # Backends report 0 or NaN for sources without a usable frame rate.
            fps = reported_fps if reported_fps and reported_fps > 0 else self.settings.runtime.source_fps_fallback
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            capture.release()
        self.stats = {"fps": fps, "width": width, "height": height, "reconnects": 0}
        return self.stats

    def frames(self, stop_event: Event | None = None) -> Iterator[FramePacket]:
        if not self.stats:
            self.probe()
        rotation = self.settings.stream.rotation
        if rotation == 0:
            yield from self._frames(stop_event)
            return
        code = _ROTATIONS[rotation]
        for packet in self._frames(stop_event):
            packet.frame = cv2.rotate(packet.frame, code)
            packet.height, packet.width = packet.frame.shape[:2]
            yield packet

    def _frames(self, stop_event: Event | None = None) -> Iterator[FramePacket]:
        stop_event = stop_event or Event()
        if self._uses_pyav():
            yield from self._frames_pyav(stop_event)
            return
        is_local_file = _is_local_file_source(self.source)
        frame_index = 0
        reconnects = 0
        fps = float(self.stats["fps"])
        width = int(self.stats["width"])
        height = int(self.stats["height"])
        stream_started_at = time.perf_counter()

        # Timestamps must never move backwards: a looped file rewinds its own
        # media clock, so each new pass continues from where the previous
        # ended. The pipeline relies on this for TTL eviction and cooldowns.
        timestamp_offset = 0.0
        timestamp = 0.0

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
                    # POS_MSEC is the presentation time of the frame just
                    # read, so the first frame legitimately reports 0.0 —
                    # the fallback must be indexed from 0, not from 1, or
                    # frames 1 and 2 collide on the same timestamp.
                    ts_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
                    if ts_ms and ts_ms > 0:
                        timestamp = ts_ms / 1000.0
                    else:
                        timestamp = (segment_frame_index - 1) / max(fps, 1.0)
                    timestamp += timestamp_offset
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
                    timestamp_offset = timestamp + 1.0 / max(fps, 1.0)
                    continue
                break
            reconnects += 1
            self.stats["reconnects"] = reconnects
            if self.settings.stream.max_reconnects >= 0 and reconnects > self.settings.stream.max_reconnects:
                break
            time.sleep(self.settings.stream.reconnect_delay_sec)

    def _frames_pyav(self, stop_event: Event) -> Iterator[FramePacket]:
        """Frames of a network stream, stamped from the media's own clock.

        Within one connection a frame's timestamp is its presentation time
        relative to the first frame of that connection, so a slow consumer
        cannot move it. A new connection starts a new segment: its clock has
        no relation to the old one, so it is placed after the last frame seen
        plus the wall time that passed, and the distance is reported as
        `gap_before` instead of being passed off as video.
        """
        fps = float(self.stats["fps"])
        width = int(self.stats["width"])
        height = int(self.stats["height"])
        reconnects = 0
        frame_index = 0
        segment_id = -1
        last_timestamp = 0.0
        last_frame_wall = 0.0
        seen_any = False

        while not stop_event.is_set():
            try:
                container = self._open_pyav()
            except Exception as exc:
                if not seen_any:
                    raise RuntimeError(f"Cannot open source: {redact_source(self.source)}") from exc
                container = None

            if container is not None:
                segment_id += 1
                base_pts_time: float | None = None
                segment_offset = 0.0
                gap_before = 0.0
                try:
                    video = container.streams.video[0]
                    time_base = video.time_base
                    for frame in container.decode(video):
                        if stop_event.is_set():
                            break
                        if frame.pts is None:
                            continue
                        pts_time = float(frame.pts * time_base)
                        if base_pts_time is None:
                            base_pts_time = pts_time
                            if seen_any:
                                segment_offset = last_timestamp + (time.monotonic() - last_frame_wall)
                                gap_before = segment_offset - last_timestamp
                        timestamp = max(segment_offset + (pts_time - base_pts_time), last_timestamp)
                        last_timestamp = timestamp
                        last_frame_wall = time.monotonic()
                        seen_any = True
                        frame_index += 1
                        yield FramePacket(
                            frame=frame.to_ndarray(format="bgr24"),
                            frame_index=frame_index,
                            timestamp=timestamp,
                            fps=fps,
                            width=width,
                            height=height,
                            segment_id=segment_id,
                            gap_before=gap_before,
                        )
                        gap_before = 0.0
                except Exception:
                    # The connection dropped or went silent past the read timeout;
                    # the segment ends here and a new one is opened below.
                    if stop_event.is_set():
                        break
                finally:
                    container.close()

            if stop_event.is_set():
                break
            reconnects += 1
            self.stats["reconnects"] = reconnects
            self.stats["segments"] = segment_id + 2
            if self.settings.stream.max_reconnects >= 0 and reconnects > self.settings.stream.max_reconnects:
                break
            stop_event.wait(self.settings.stream.reconnect_delay_sec)
