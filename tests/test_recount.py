"""Tests for recount.py — lap counting from an event stream.

The key invariant: laps are derived purely from the sorted event stream.
Events written out of order (because reid-watch appends after detect) must
still produce correct lap numbers and lap_times.
"""
import csv
import json
import tempfile
from pathlib import Path

import pytest

from mx_tracker.recount import recount


def _write_jsonl(path: Path, events: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")


def _read_results(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


class TestRecountBasic:
    def test_single_rider_single_lap(self, tmp_path):
        _write_jsonl(tmp_path / "events.jsonl", [
            {"timestamp": 10.0, "rider_id": "plate_133", "identity_source": "plate",
             "frame_index": 1, "tracker_id": 1, "plate_text": "133", "plate_conf": 0.9,
             "lap": "", "lap_time": "", "bbox": [0,0,50,50], "center": [25,25], "crop_file": ""},
        ])
        recount(tmp_path)
        rows = _read_results(tmp_path / "results.csv")
        assert len(rows) == 1
        assert rows[0]["rider_id"] == "plate_133"
        assert rows[0]["lap"] == "1"
        assert float(rows[0]["lap_time"]) == pytest.approx(10.0)  # ts - race_start_sec(0.0)

    def test_single_rider_two_laps_computes_lap_time(self, tmp_path):
        _write_jsonl(tmp_path / "events.jsonl", [
            {"timestamp": 10.0, "rider_id": "plate_133", "identity_source": "plate",
             "frame_index": 1, "tracker_id": 1, "plate_text": "133", "plate_conf": 0.9,
             "lap": "", "lap_time": "", "bbox": [0,0,50,50], "center": [25,25], "crop_file": ""},
            {"timestamp": 75.5, "rider_id": "plate_133", "identity_source": "plate",
             "frame_index": 100, "tracker_id": 1, "plate_text": "133", "plate_conf": 0.95,
             "lap": "", "lap_time": "", "bbox": [0,0,50,50], "center": [25,25], "crop_file": ""},
        ])
        recount(tmp_path)
        rows = _read_results(tmp_path / "results.csv")
        assert rows[0]["lap"] == "1"
        assert float(rows[0]["lap_time"]) == pytest.approx(10.0)  # ts - race_start_sec(0.0)
        assert rows[1]["lap"] == "2"
        assert float(rows[1]["lap_time"]) == pytest.approx(65.5)

    def test_unresolved_events_excluded_from_results(self, tmp_path):
        _write_jsonl(tmp_path / "events.jsonl", [
            {"timestamp": 10.0, "rider_id": "plate_133", "identity_source": "plate",
             "frame_index": 1, "tracker_id": 1, "plate_text": "133", "plate_conf": 0.9,
             "lap": "", "lap_time": "", "bbox": [0,0,50,50], "center": [25,25], "crop_file": ""},
            {"timestamp": 15.0, "rider_id": "", "identity_source": "unresolved",
             "frame_index": 2, "tracker_id": 2, "plate_text": "", "plate_conf": 0.0,
             "lap": "", "lap_time": "", "bbox": [0,0,50,50], "center": [25,25], "crop_file": ""},
        ])
        recount(tmp_path)
        rows = _read_results(tmp_path / "results.csv")
        assert len(rows) == 1
        assert rows[0]["rider_id"] == "plate_133"

    def test_two_riders_counted_independently(self, tmp_path):
        _write_jsonl(tmp_path / "events.jsonl", [
            {"timestamp": 10.0, "rider_id": "plate_133", "identity_source": "plate",
             "frame_index": 1, "tracker_id": 1, "plate_text": "133", "plate_conf": 0.9,
             "lap": "", "lap_time": "", "bbox": [0,0,50,50], "center": [25,25], "crop_file": ""},
            {"timestamp": 11.0, "rider_id": "plate_27", "identity_source": "plate",
             "frame_index": 2, "tracker_id": 2, "plate_text": "27", "plate_conf": 0.85,
             "lap": "", "lap_time": "", "bbox": [0,0,50,50], "center": [25,25], "crop_file": ""},
            {"timestamp": 80.0, "rider_id": "plate_133", "identity_source": "plate",
             "frame_index": 100, "tracker_id": 1, "plate_text": "133", "plate_conf": 0.9,
             "lap": "", "lap_time": "", "bbox": [0,0,50,50], "center": [25,25], "crop_file": ""},
            {"timestamp": 85.0, "rider_id": "plate_27", "identity_source": "plate",
             "frame_index": 110, "tracker_id": 2, "plate_text": "27", "plate_conf": 0.85,
             "lap": "", "lap_time": "", "bbox": [0,0,50,50], "center": [25,25], "crop_file": ""},
        ])
        recount(tmp_path)
        rows = _read_results(tmp_path / "results.csv")
        by_rider = {r["rider_id"]: [] for r in rows}
        for r in rows:
            by_rider[r["rider_id"]].append(r)

        assert by_rider["plate_133"][0]["lap"] == "1"
        assert by_rider["plate_133"][1]["lap"] == "2"
        assert by_rider["plate_27"][0]["lap"] == "1"
        assert by_rider["plate_27"][1]["lap"] == "2"

        # lap_time for rider 27: 85-11 = 74s
        assert float(by_rider["plate_27"][1]["lap_time"]) == pytest.approx(74.0)

    def test_events_appended_out_of_order_sorted_correctly(self, tmp_path):
        # detect writes t=10 and t=80. reid-watch later appends t=45 (resolved unresolved).
        # recount must sort by timestamp → lap at t=45 must be lap 2, not lap 3.
        _write_jsonl(tmp_path / "events.jsonl", [
            {"timestamp": 10.0, "rider_id": "plate_133", "identity_source": "plate",
             "frame_index": 1, "tracker_id": 1, "plate_text": "133", "plate_conf": 0.9,
             "lap": "", "lap_time": "", "bbox": [0,0,50,50], "center": [25,25], "crop_file": ""},
            {"timestamp": 80.0, "rider_id": "plate_133", "identity_source": "plate",
             "frame_index": 100, "tracker_id": 1, "plate_text": "133", "plate_conf": 0.9,
             "lap": "", "lap_time": "", "bbox": [0,0,50,50], "center": [25,25], "crop_file": ""},
            # appended later by reid-watch — chronologically it's the second crossing
            {"timestamp": 45.0, "rider_id": "plate_133", "identity_source": "manual",
             "frame_index": 50, "tracker_id": 3, "plate_text": "133", "plate_conf": 0.0,
             "lap": "", "lap_time": "", "bbox": [0,0,50,50], "center": [25,25], "crop_file": ""},
        ])
        recount(tmp_path)
        rows = _read_results(tmp_path / "results.csv")
        laps = [(float(r["timestamp"]), int(r["lap"])) for r in rows if r["rider_id"] == "plate_133"]
        assert laps == [(10.0, 1), (45.0, 2), (80.0, 3)]

    def test_all_identity_sources_except_unresolved_included(self, tmp_path):
        sources = ["plate", "reid", "plate_reread", "reid_post", "manual"]
        events = [
            {"timestamp": float(i), "rider_id": f"plate_{i}", "identity_source": src,
             "frame_index": i, "tracker_id": i, "plate_text": str(i), "plate_conf": 0.9,
             "lap": "", "lap_time": "", "bbox": [0,0,50,50], "center": [25,25], "crop_file": ""}
            for i, src in enumerate(sources, start=1)
        ]
        _write_jsonl(tmp_path / "events.jsonl", events)
        recount(tmp_path)
        rows = _read_results(tmp_path / "results.csv")
        assert len(rows) == len(sources)

    def test_raises_when_jsonl_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            recount(tmp_path)

    def test_results_csv_created_in_run_dir(self, tmp_path):
        _write_jsonl(tmp_path / "events.jsonl", [
            {"timestamp": 5.0, "rider_id": "plate_9", "identity_source": "plate",
             "frame_index": 1, "tracker_id": 1, "plate_text": "9", "plate_conf": 0.8,
             "lap": "", "lap_time": "", "bbox": [0,0,50,50], "center": [25,25], "crop_file": ""},
        ])
        out = recount(tmp_path)
        assert out == tmp_path / "results.csv"
        assert out.exists()
