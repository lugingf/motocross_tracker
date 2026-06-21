"""Recount laps from events.jsonl and produce results.csv.

Reads all resolved crossings, sorts by timestamp, assigns lap numbers
and lap_times per rider. Unresolved crossings are excluded.

Usage:
    mx-tracker recount --run-dir data/artifacts/detect_file_20240119_120000
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

Logger = Callable[[str], None]

_RESULT_FIELDS = [
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


def _flatten(event: dict) -> dict:
    """Expand center list to individual columns; drop bbox."""
    out = dict(event)
    out.pop("bbox", None)
    center = out.pop("center", None)
    if center is not None:
        out["center_x"], out["center_y"] = center
    return out


def _read_started_at(run_dir: Path) -> datetime | None:
    info_path = run_dir / "run_info.json"
    if not info_path.exists():
        return None
    try:
        data = json.loads(info_path.read_text(encoding="utf-8"))
        return datetime.fromisoformat(data["started_at"])
    except Exception:
        return None


def recount(
    run_dir: str | Path,
    logger: Logger | None = None,
    race_start_sec: float = 0.0,
    race_start_at: str | None = None,
) -> Path:
    log = logger or print
    run_dir = Path(run_dir)
    jsonl_path = run_dir / "events.jsonl"
    out_path = run_dir / "results.csv"

    if not jsonl_path.exists():
        raise FileNotFoundError(f"events.jsonl not found: {jsonl_path}")

    started_at = _read_started_at(run_dir)

    # If race_start_at (wall-clock ISO time) provided, convert to seconds offset
    if race_start_at is not None and started_at is not None:
        try:
            race_start_dt = datetime.fromisoformat(race_start_at)
            race_start_sec = (race_start_dt - started_at).total_seconds()
        except Exception:
            log(f"[recount] warning: could not parse race_start_at={race_start_at!r}, using race_start_sec={race_start_sec}")

    # Read all resolved events
    resolved: list[dict] = []
    with jsonl_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("identity_source") == "unresolved":
                continue
            if not r.get("rider_id"):
                continue
            try:
                r["_ts"] = float(r["timestamp"])
            except (KeyError, TypeError, ValueError):
                continue
            resolved.append(r)

    # Sort by timestamp
    resolved.sort(key=lambda r: r["_ts"])

    # Assign lap and lap_time per rider; lap 1 is measured from race_start_sec
    last_ts: dict[str, float] = {}
    lap_count: dict[str, int] = {}
    rows: list[dict] = []
    for r in resolved:
        rider_id = r["rider_id"]
        ts = r["_ts"]
        lap_count[rider_id] = lap_count.get(rider_id, 0) + 1
        lap = lap_count[rider_id]
        prev = last_ts.get(rider_id, race_start_sec)
        lap_time = round(ts - prev, 3)
        last_ts[rider_id] = ts

        row = _flatten(r)
        row.pop("_ts", None)
        row["lap"] = lap
        row["lap_time"] = lap_time
        if "wall_time" not in row and started_at is not None:
            row["wall_time"] = (started_at + timedelta(seconds=ts)).isoformat(timespec="seconds")
        rows.append(row)

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_RESULT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    log(f"[recount] {len(rows)} events → {out_path} (race_start_sec={race_start_sec})")
    log(f"[recount] riders: {len(lap_count)}, total crossings: {sum(lap_count.values())}")
    for rider_id, laps in sorted(lap_count.items(), key=lambda x: -x[1]):
        log(f"  {rider_id}: {laps} lap(s)")

    return out_path
