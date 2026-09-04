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


# ---------------------------------------------------------------------------
# race_start_at — the wall-clock alternative to race_start_sec.
#
# configs/recount.yaml and the CLI help both document a bare time of day
# ("10:31:00"). That form used to fail parsing and silently fall back to
# race_start_sec=0, so lap 1 was measured from the start of the video.
# ---------------------------------------------------------------------------

def _run_with_start(tmp_path: Path, started_at: str, events: list[dict]) -> None:
    (tmp_path / "run_info.json").write_text(
        json.dumps({"started_at": started_at, "source": "race.mp4"}), encoding="utf-8"
    )
    _write_jsonl(tmp_path / "events.jsonl", events)


def _event(ts: float, rider: str = "plate_133") -> dict:
    return {
        "timestamp": ts, "rider_id": rider, "identity_source": "plate",
        "frame_index": int(ts * 30), "tracker_id": 1, "plate_text": rider.removeprefix("plate_"),
        "plate_conf": 0.9, "lap": "", "lap_time": "",
        "bbox": [0, 0, 50, 50], "center": [25, 25], "crop_file": "",
    }


class TestRaceStartAt:
    STARTED = "2026-06-21T10:30:00+04:00"

    def test_bare_time_of_day_offsets_lap_one(self, tmp_path):
        # Video starts 10:30:00, race starts 10:31:00 → 60 s in.
        # A crossing at t=100 s is 40 s into lap 1, not 100 s.
        _run_with_start(tmp_path, self.STARTED, [_event(100.0)])
        recount(tmp_path, race_start_at="10:31:00")
        rows = _read_results(tmp_path / "results.csv")
        assert float(rows[0]["lap_time"]) == pytest.approx(40.0)

    def test_bare_time_without_seconds_accepted(self, tmp_path):
        _run_with_start(tmp_path, self.STARTED, [_event(100.0)])
        recount(tmp_path, race_start_at="10:31")
        rows = _read_results(tmp_path / "results.csv")
        assert float(rows[0]["lap_time"]) == pytest.approx(40.0)

    def test_full_iso_with_offset_accepted(self, tmp_path):
        _run_with_start(tmp_path, self.STARTED, [_event(100.0)])
        recount(tmp_path, race_start_at="2026-06-21T10:31:00+04:00")
        rows = _read_results(tmp_path / "results.csv")
        assert float(rows[0]["lap_time"]) == pytest.approx(40.0)

    def test_naive_iso_datetime_inherits_run_timezone(self, tmp_path):
        _run_with_start(tmp_path, self.STARTED, [_event(100.0)])
        recount(tmp_path, race_start_at="2026-06-21T10:31:00")
        rows = _read_results(tmp_path / "results.csv")
        assert float(rows[0]["lap_time"]) == pytest.approx(40.0)

    def test_naive_run_info_accepts_naive_race_start(self, tmp_path):
        _run_with_start(tmp_path, "2026-06-21T10:30:00", [_event(100.0)])
        recount(tmp_path, race_start_at="10:31:00")
        rows = _read_results(tmp_path / "results.csv")
        assert float(rows[0]["lap_time"]) == pytest.approx(40.0)

    def test_only_lap_one_is_measured_from_race_start(self, tmp_path):
        _run_with_start(tmp_path, self.STARTED, [_event(100.0), _event(220.0)])
        recount(tmp_path, race_start_at="10:31:00")
        rows = _read_results(tmp_path / "results.csv")
        assert [float(r["lap_time"]) for r in rows] == [
            pytest.approx(40.0), pytest.approx(120.0)
        ]

    def test_race_start_at_overrides_race_start_sec(self, tmp_path):
        _run_with_start(tmp_path, self.STARTED, [_event(100.0)])
        recount(tmp_path, race_start_sec=10.0, race_start_at="10:31:00")
        rows = _read_results(tmp_path / "results.csv")
        assert float(rows[0]["lap_time"]) == pytest.approx(40.0)

    def test_unparsable_value_warns_and_falls_back(self, tmp_path):
        _run_with_start(tmp_path, self.STARTED, [_event(100.0)])
        messages: list[str] = []
        recount(tmp_path, logger=messages.append, race_start_sec=10.0, race_start_at="not a time")
        rows = _read_results(tmp_path / "results.csv")
        assert float(rows[0]["lap_time"]) == pytest.approx(90.0)
        assert any("could not parse" in m for m in messages)

    def test_missing_run_info_warns_that_it_is_required(self, tmp_path):
        _write_jsonl(tmp_path / "events.jsonl", [_event(100.0)])
        messages: list[str] = []
        recount(tmp_path, logger=messages.append, race_start_at="10:31:00")
        assert any("run_info.json" in m for m in messages)
