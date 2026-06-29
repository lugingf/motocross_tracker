"""Video overlay rendering.

`OverlayWriter` owns the output `VideoWriter` and knows how to draw the finish
line, per-vehicle labels, the live leaderboard and recent-crossing banners. The
pipeline hands it state each frame; all drawing logic lives here.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import TrackerSettings
from .tracking import DetectionRecord
from .video_source import FramePacket


def _draw_outlined_text(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    font_scale: float = 0.7,
    thickness: int = 2,
) -> None:
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 4, cv2.LINE_AA)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


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


class OverlayWriter:
    """Draws the per-frame overlay and writes it to an MP4."""

    def __init__(
        self,
        video_path: Path | None,
        settings: TrackerSettings,
        line_a: tuple[int, int],
        line_b: tuple[int, int],
        fps: float,
        width: int,
        height: int,
        collect_only: bool,
    ) -> None:
        self.settings = settings
        self.line_a = line_a
        self.line_b = line_b
        self.collect_only = collect_only
        self._writer: cv2.VideoWriter | None = None
        if video_path is not None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(str(video_path), fourcc, max(fps, 1.0), (width, height))

    @property
    def enabled(self) -> bool:
        return self._writer is not None

    def render(
        self,
        frame: np.ndarray,
        packet: FramePacket,
        records: list[DetectionRecord],
        crossing_counts: dict[str, int],
        recent_crossings: dict[str, tuple[str, float]],
    ) -> None:
        if self._writer is None:
            return
        cv2.line(frame, self.line_a, self.line_b, (255, 255, 255), self.settings.line.width)
        for record in records:
            label = f"tid={record.tracker_id}"
            if record.plate_text:
                label += f" plate={record.plate_text}:{record.plate_conf:.2f}"
            _draw_label(frame, record.bbox, label)
        if not self.collect_only:
            self._draw_leaderboard(frame, crossing_counts)
            if self.settings.output.overlay_crossing_text:
                self._draw_crossing_banner(frame, packet, recent_crossings)
        self._writer.write(frame)

    def _draw_leaderboard(self, frame: np.ndarray, crossing_counts: dict[str, int]) -> None:
        top_rows = sorted(crossing_counts.items(), key=lambda item: (-item[1], item[0]))[
            : self.settings.output.overlay_top_n
        ]
        for index, (rider_id, count) in enumerate(top_rows):
            _draw_outlined_text(frame, f"{rider_id} x{count}", (16, 36 + index * 28))

    def _draw_crossing_banner(
        self,
        frame: np.ndarray,
        packet: FramePacket,
        recent_crossings: dict[str, tuple[str, float]],
    ) -> None:
        display_sec = self.settings.line.cooldown_sec
        active = sorted(
            ((txt, ts) for txt, ts in recent_crossings.values() if packet.timestamp - ts <= display_sec),
            key=lambda item: item[1],
        )
        h, w = frame.shape[:2]
        font_scale = max(1.5, w / 800.0)
        for slot, (txt, _) in enumerate(active):
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 3)
            x = (w - tw) // 2
            y = h - 32 - slot * int(th * 1.5)
            _draw_outlined_text(frame, txt, (x, y), font_scale=font_scale, thickness=3)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
