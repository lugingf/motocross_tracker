"""Lap calculation from line crossings.

One place decides what a lap is, so that a recount of a finished run and the
live import of a stream give the same laps for the same crossings.

A rider's first crossing is a lap when the race start is known: it is measured
from the start. When the start is unknown it is only a baseline, and the next
crossing is the first lap that was measured in full.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable, Sequence

QUALITY_GAP_BEFORE = "gap_before"
QUALITY_LOW_CONFIDENCE = "low_confidence"


@dataclass(frozen=True, slots=True)
class Crossing:
    """One pass over the line. `key` is whatever the caller uses to find it again."""

    key: Hashable
    timestamp: float
    participant_id: str
    number: int | None = None
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class Gap:
    """A stretch of the timeline for which no video was seen."""

    start: float
    end: float


@dataclass(slots=True)
class Lap:
    participant_id: str
    number: int | None
    lap_number: int
    lap_time: float
    finish_ts: float
    position: int
    confidence: float | None
    quality: list[str] = field(default_factory=list)
    keys: list[Hashable] = field(default_factory=list)


@dataclass(slots=True)
class LapResult:
    laps: list[Lap]
    assignments: dict[Hashable, tuple[int, float]]
    baselines: list[Hashable]
    ignored: list[Hashable]


def compute_laps(
    crossings: Sequence[Crossing],
    race_start: float | None,
    min_lap: float = 0.0,
    gaps: Sequence[Gap] = (),
    low_confidence: float = 0.0,
) -> LapResult:
    """Turn crossings into laps.

    `race_start` is the timeline position of the start, or None when it is
    not known. A crossing closer than `min_lap` to the previous accepted one
    of the same rider is a duplicate and is ignored; it does not move that
    rider's clock. A lap that spans a gap in the video is flagged, never
    mended.
    """
    ordered = sorted(enumerate(crossings), key=lambda item: (item[1].timestamp, item[0]))

    last_ts: dict[str, float] = {}
    lap_count: dict[str, int] = {}
    standing: dict[str, tuple[int, float]] = {}
    laps: list[Lap] = []
    assignments: dict[Hashable, tuple[int, float]] = {}
    baselines: list[Hashable] = []
    ignored: list[Hashable] = []

    for _, crossing in ordered:
        rider = crossing.participant_id
        ts = crossing.timestamp
        previous = last_ts.get(rider, race_start)

        if previous is None:
            last_ts[rider] = ts
            baselines.append(crossing.key)
            continue
        if min_lap > 0 and (ts - previous) < min_lap:
            ignored.append(crossing.key)
            continue

        lap_count[rider] = lap_count.get(rider, 0) + 1
        lap_number = lap_count[rider]
        lap_time = round(ts - previous, 3)
        last_ts[rider] = ts
        standing[rider] = (lap_number, ts)

        quality: list[str] = []
        if any(gap.start < ts and gap.end > previous for gap in gaps):
            quality.append(QUALITY_GAP_BEFORE)
        if low_confidence > 0 and crossing.confidence is not None and crossing.confidence < low_confidence:
            quality.append(QUALITY_LOW_CONFIDENCE)

        position = 1 + sum(1 for other, (laps_done, at) in standing.items() if other != rider and (laps_done > lap_number or (laps_done == lap_number and at < ts)))
        laps.append(
            Lap(
                participant_id=rider,
                number=crossing.number,
                lap_number=lap_number,
                lap_time=lap_time,
                finish_ts=ts,
                position=position,
                confidence=crossing.confidence,
                quality=quality,
                keys=[crossing.key],
            )
        )
        assignments[crossing.key] = (lap_number, lap_time)

    return LapResult(laps=laps, assignments=assignments, baselines=baselines, ignored=ignored)
