"""From stored events to a revision of the laps.

The same `compute_laps` that recounts a finished run decides what a lap is;
this module only prepares its input from what lap_vision knows — the entry
list, the race start and the corrections a person made — and shapes its output
for the contract.
"""
from __future__ import annotations

from typing import Any

from ..laps import Crossing, Gap, compute_laps

ALGORITHM_VERSION = "laps-1"
SCHEMA_VERSION = 1


def number_of(participant_id: str) -> int | None:
    prefix = "plate_"
    if participant_id.startswith(prefix) and participant_id[len(prefix):].isdigit():
        return int(participant_id[len(prefix):])
    return None


def _participant(number: int) -> str:
    return f"plate_{number}"


def apply_corrections(crossings: list[dict[str, Any]], corrections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the crossings as the person has corrected them, without touching the originals.

    Corrections apply in the order they were made, so a later one can undo an
    earlier one. A crossing that stays excluded is dropped.
    """
    adjusted = {item["event_id"]: dict(item) for item in crossings}
    excluded: set[str] = set()

    for correction in sorted(corrections, key=lambda c: c["id"]):
        kind = correction.get("kind")
        ids = correction.get("event_ids") or []
        if kind == "set_number":
            for event_id in ids:
                if event_id in adjusted:
                    adjusted[event_id]["participant_id"] = _participant(int(correction["number"]))
                    adjusted[event_id]["identity_source"] = "manual"
        elif kind == "merge":
            source = _participant(int(correction["from_number"]))
            target = _participant(int(correction["number"]))
            for item in adjusted.values():
                if item["participant_id"] == source:
                    item["participant_id"] = target
                    item["identity_source"] = "manual"
        elif kind == "exclude":
            excluded.update(ids)
        elif kind == "include":
            excluded.difference_update(ids)

    return [item for event_id, item in adjusted.items() if event_id not in excluded]


def build_revision(
    events: list[dict[str, Any]],
    corrections: list[dict[str, Any]],
    configuration: dict[str, Any],
    race_start_ms: int | None,
    low_confidence: float,
    run_id: str,
    submission_id: str,
    status: str,
    applied_correction_id: int,
) -> dict[str, Any]:
    """Calculate a revision from `events`, which must be a gap-free run of sequence numbers."""
    calibration = configuration.get("calibration") or {}
    roster = {entry["number"]: entry for entry in configuration.get("roster") or []}
    min_lap = float(calibration.get("min_lap_time_ms", 0)) / 1000.0

    crossings = [event for event in events if event["kind"] == "crossing"]
    gaps = [
        Gap(start=event["media_time_ms"] / 1000.0, end=(event.get("extra") or {}).get("to_ms", event["media_time_ms"]) / 1000.0)
        for event in events
        if event["kind"] == "gap"
    ]

    counted: list[Crossing] = []
    for item in apply_corrections(crossings, corrections):
        participant = item["participant_id"]
        if not participant:
            continue
        counted.append(
            Crossing(
                key=item["event_id"],
                timestamp=item["media_time_ms"] / 1000.0,
                participant_id=participant,
                number=number_of(participant),
                confidence=item.get("plate_confidence"),
            )
        )

    result = compute_laps(
        counted,
        race_start=None if race_start_ms is None else race_start_ms / 1000.0,
        min_lap=min_lap,
        gaps=gaps,
        low_confidence=low_confidence,
    )

    laps = []
    for lap in result.laps:
        entry = roster.get(lap.number) if lap.number is not None else None
        quality = list(lap.quality) + (["unknown_number"] if lap.number is None else [])
        laps.append(
            {
                "participant_id": lap.participant_id,
                "number": lap.number,
                "driver_name": entry["name"] if entry else "",
                "team": entry.get("team", "") if entry else "",
                "lap_number": lap.lap_number,
                "lap_time_ms": round(lap.lap_time * 1000),
                "finish_media_ms": round(lap.finish_ts * 1000),
                "position": lap.position,
                "confidence": lap.confidence,
                "quality": quality,
                "event_ids": lap.keys,
            }
        )

    last_seq = events[-1]["seq"] if events else 0
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "submission_id": submission_id,
        "status": status,
        "algorithm_version": ALGORITHM_VERSION,
        "config_version": int(configuration.get("version", 0)),
        "race_start_media_ms": race_start_ms,
        "manifest": {
            "last_seq": last_seq,
            "event_count": len(events),
            "gaps": [
                {"from_ms": round(gap.start * 1000), "to_ms": round(gap.end * 1000), "reason": "reconnect"} for gap in gaps
            ],
        },
        "laps": laps,
        "applied_correction_id": applied_correction_id,
    }
