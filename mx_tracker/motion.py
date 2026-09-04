"""Cheap motion gate over the frame's entry zone.

Full-frame vehicle tracking dominates the per-frame cost — about 31 ms of a
50 ms frame on a 2704x1520 source — and a session holds long stretches with no
bike in sight. The gate answers one much cheaper question per frame ("has
anything moved where riders come in?") by differencing a downscaled grayscale
crop of the entry strip against a running background, and lets the pipeline
skip detection entirely when the answer is no.

Skipping is only safe while nothing is in play, so a frame is processed when
any of these holds:

* the background is still settling (`warmup_frames`),
* the entry strip changed by at least `min_area_fraction` of its pixels,
* a bike was detected within the last `hold_sec` — the pipeline reports this
  through `note_detection`, which is what keeps a rider that is already
  mid-frame from being dropped once it leaves the entry strip.

The gate degrades to a no-op rather than guessing: if the entry zone cannot be
placed unambiguously, it disables itself and every frame is processed.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import TrackerSettings
from .geometry import entry_zone_rect


@dataclass(slots=True)
class GateDecision:
    """Why a frame is being processed, or why it is being skipped."""

    process: bool
    reason: str
    change_fraction: float


class MotionGate:
    """Decides, per frame, whether detection needs to run at all."""

    def __init__(
        self,
        settings: TrackerSettings,
        line_a: tuple[int, int],
        line_b: tuple[int, int],
        frame_width: int,
        frame_height: int,
    ) -> None:
        gate = settings.motion_gate
        self.enabled = gate.enabled
        self.downscale = max(1, gate.downscale)
        self.pixel_delta = float(gate.pixel_delta)
        self.min_area_fraction = float(gate.min_area_fraction)
        self.hold_sec = float(gate.hold_sec)
        self.warmup_frames = max(0, gate.warmup_frames)
        self.background_alpha = float(gate.background_alpha)

        self.zone: tuple[int, int, int, int] | None = None
        self.disabled_reason: str | None = None
        if not self.enabled:
            self.disabled_reason = "disabled in settings"
        else:
            zone = entry_zone_rect(
                line_a, line_b, settings.line.direction, frame_width, frame_height, gate.zone_fraction
            )
            if zone is None:
                self.enabled = False
                self.disabled_reason = "entry zone is ambiguous for this finish line"
            else:
                self.zone = zone

        self.frames_seen = 0
        self.processed = 0
        self.gated = 0
        self._background: np.ndarray | None = None
        self._last_evidence_ts: float | None = None

    # -- reporting ---------------------------------------------------------

    @property
    def active(self) -> bool:
        return self.enabled and self.zone is not None

    def describe(self) -> str:
        if not self.active:
            return f"motion_gate=off ({self.disabled_reason})"
        x1, y1, x2, y2 = self.zone
        return (
            f"motion_gate=on zone={x1},{y1},{x2},{y2} "
            f"min_area={self.min_area_fraction} hold={self.hold_sec}s"
        )

    def stats(self) -> dict[str, object]:
        total = self.processed + self.gated
        return {
            "enabled": self.active,
            "zone": list(self.zone) if self.zone is not None else None,
            "processed_frames": self.processed,
            "gated_frames": self.gated,
            "gated_fraction": round(self.gated / total, 4) if total else 0.0,
            "disabled_reason": self.disabled_reason,
        }

    # -- per-frame ---------------------------------------------------------

    def note_detection(self, timestamp: float) -> None:
        """Record that a bike was seen, holding the gate open around it."""
        self._last_evidence_ts = timestamp

    def should_process(self, frame: np.ndarray, timestamp: float) -> GateDecision:
        if not self.active:
            self.processed += 1
            return GateDecision(True, "disabled", 0.0)

        patch = self._patch(frame)
        if patch is None:
            self.processed += 1
            return GateDecision(True, "no_zone", 0.0)

        self.frames_seen += 1
        fraction = self._change_fraction(patch)
        self._update_background(patch)

        decision = self._decide(fraction, timestamp)
        if decision.process:
            self.processed += 1
        else:
            self.gated += 1
        return decision

    def _decide(self, fraction: float, timestamp: float) -> GateDecision:
        if self.frames_seen <= self.warmup_frames:
            return GateDecision(True, "warmup", fraction)
        if fraction >= self.min_area_fraction:
            self._last_evidence_ts = timestamp
            return GateDecision(True, "motion", fraction)
        if self._last_evidence_ts is not None and (timestamp - self._last_evidence_ts) <= self.hold_sec:
            return GateDecision(True, "hold", fraction)
        return GateDecision(False, "idle", fraction)

    # -- internals ---------------------------------------------------------

    def _patch(self, frame: np.ndarray) -> np.ndarray | None:
        x1, y1, x2, y2 = self.zone
        strip = frame[y1:y2, x1:x2]
        if strip.size == 0:
            return None
        if strip.ndim == 3:
            strip = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
        if self.downscale > 1:
            height = max(1, strip.shape[0] // self.downscale)
            width = max(1, strip.shape[1] // self.downscale)
            strip = cv2.resize(strip, (width, height), interpolation=cv2.INTER_AREA)
        return strip.astype(np.float32, copy=False)

    def _change_fraction(self, patch: np.ndarray) -> float:
        if self._background is None or self._background.shape != patch.shape:
            # Nothing to compare against yet: treat the frame as fully changed
            # so it gets processed instead of skipped on no evidence.
            return 1.0
        changed = np.count_nonzero(np.abs(patch - self._background) >= self.pixel_delta)
        return changed / patch.size

    def _update_background(self, patch: np.ndarray) -> None:
        if self._background is None or self._background.shape != patch.shape:
            self._background = patch.copy()
            return
        alpha = self.background_alpha
        self._background += alpha * (patch - self._background)
