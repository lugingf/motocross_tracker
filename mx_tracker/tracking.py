"""Per-track state for vehicles followed across frames.

A `TrackState` accumulates plate observations for a single tracker id and
decides, at crossing time, which plate number that rider most likely carries.
This module holds only in-memory domain state — no models, no I/O.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field


@dataclass(slots=True)
class PlateObservation:
    """A single plate read for a track at a given moment."""

    text: str
    confidence: float
    timestamp: float
    frame_index: int


@dataclass(slots=True)
class DetectionRecord:
    """A vehicle detection in one frame, used for overlay rendering."""

    tracker_id: int
    bbox: tuple[int, int, int, int]
    center: tuple[int, int]
    plate_text: str | None = None
    plate_conf: float = 0.0


@dataclass(slots=True)
class TrackState:
    """Rolling history of one tracked vehicle.

    Keeps recent plate observations within a TTL window and votes on the most
    likely plate number when the track crosses the finish line.
    """

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
        """Most frequent plate within the vote window, tie-broken by score."""
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
