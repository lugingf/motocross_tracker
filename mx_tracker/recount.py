"""Recount laps from events.jsonl and produce results.csv.

Reads all resolved crossings, sorts by timestamp, assigns lap numbers
and lap_times per rider. Unresolved crossings are excluded.

Usage:
    mx-tracker recount --run-dir data/artifacts/detect_file_20240119_120000
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Callable

Logger = Callable[[str], None]

_RESULT_FIELDS = [
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


def _flatten(event: dict) -> dict:
    """Expand bbox/center lists to individual columns if present."""
    out = dict(event)
    bbox = out.pop("bbox", None)
    center = out.pop("center", None)
    if bbox is not None:
        out["bbox_x1"], out["bbox_y1"], out["bbox_x2"], out["bbox_y2"] = bbox
    if center is not None:
        out["center_x"], out["center_y"] = center
    return out


def recount(run_dir: str | Path, logger: Logger | None = None) -> Path:
    log = logger or print
    run_dir = Path(run_dir)
    jsonl_path = run_dir / "events.jsonl"
    out_path = run_dir / "results.csv"

    if not jsonl_path.exists():
        raise FileNotFoundError(f"events.jsonl not found: {jsonl_path}")

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

    # Assign lap and lap_time per rider
    last_ts: dict[str, float] = {}
    lap_count: dict[str, int] = {}
    rows: list[dict] = []
    for r in resolved:
        rider_id = r["rider_id"]
        ts = r["_ts"]
        lap_count[rider_id] = lap_count.get(rider_id, 0) + 1
        lap = lap_count[rider_id]
        prev = last_ts.get(rider_id)
        lap_time = round(ts - prev, 3) if prev is not None else ""
        last_ts[rider_id] = ts

        row = _flatten(r)
        row.pop("_ts", None)
        row["lap"] = lap
        row["lap_time"] = lap_time
        rows.append(row)

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_RESULT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    log(f"[recount] {len(rows)} events → {out_path}")
    log(f"[recount] riders: {len(lap_count)}, total crossings: {sum(lap_count.values())}")
    for rider_id, laps in sorted(lap_count.items(), key=lambda x: -x[1]):
        log(f"  {rider_id}: {laps} lap(s)")

    return out_path
