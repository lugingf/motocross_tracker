"""Tests for artifacts.py — run layout, the event log and the crop archive.

The event log is the run's only durable record: recount and reid-watch both
read it back, so the JSONL and CSV views must stay in agreement and must
contain exactly the columns those consumers expect.
"""
import csv
import json
from datetime import datetime, timezone

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from mx_tracker.artifacts import (
    COLLECT_CSV_FIELDS,
    EVENT_CSV_FIELDS,
    CropArchive,
    EventLog,
    prepare_artifacts,
)
from mx_tracker.plate_reading import PlateRead
from mx_tracker.tracking import PlateObservation, TrackState


STARTED_AT = datetime(2026, 6, 21, 10, 30, 0, tzinfo=timezone.utc)


def _artifacts(tmp_path, **flags):
    options = {
        "write_video": False,
        "write_csv": True,
        "write_jsonl": True,
        "write_summary": False,
        "save_plate_crops": False,
    }
    options.update(flags)
    return prepare_artifacts(output_dir=tmp_path, prefix="detect_file", **options)


def _crossing(**overrides):
    event = {
        "timestamp": 12.5,
        "frame_index": 375,
        "tracker_id": 7,
        "rider_id": "plate_133",
        "identity_source": "plate",
        "plate_text": "133",
        "plate_conf": 0.876543,
        "bbox": (10, 20, 110, 220),
        "center": (60, 120),
        "crop_file": "",
    }
    event.update(overrides)
    return event


def _read_csv(path):
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# prepare_artifacts — the output flags decide which files exist at all
# ---------------------------------------------------------------------------

class TestPrepareArtifacts:
    def test_uses_the_requested_output_dir(self, tmp_path):
        target = tmp_path / "run_01"
        assert _artifacts(target).run_dir == target.resolve()

    def test_creates_the_run_dir(self, tmp_path):
        assert _artifacts(tmp_path / "nested" / "run").run_dir.is_dir()

    def test_existing_run_dir_is_reused(self, tmp_path):
        (tmp_path / "run").mkdir()
        (tmp_path / "run" / "keep.txt").write_text("x")
        _artifacts(tmp_path / "run")
        assert (tmp_path / "run" / "keep.txt").exists()

    def test_default_output_dir_lands_under_data_artifacts(self, tmp_path):
        from mx_tracker.runtime import REPO_ROOT

        artifacts = prepare_artifacts(
            output_dir=None, prefix="unit_test_prefix", write_video=False, write_csv=False,
            write_jsonl=False, write_summary=False, save_plate_crops=False,
        )
        try:
            assert artifacts.run_dir.parent == (REPO_ROOT / "data" / "artifacts").resolve()
            assert artifacts.run_dir.name.startswith("unit_test_prefix_")
        finally:
            artifacts.run_dir.rmdir()

    def test_disabled_outputs_are_none(self, tmp_path):
        artifacts = _artifacts(tmp_path, write_csv=False, write_jsonl=False)
        assert artifacts.csv_path is None
        assert artifacts.jsonl_path is None
        assert artifacts.video_path is None
        assert artifacts.summary_path is None

    def test_enabled_outputs_get_conventional_names(self, tmp_path):
        artifacts = _artifacts(tmp_path, write_video=True, write_summary=True)
        assert artifacts.video_path.name == "overlay.mp4"
        assert artifacts.csv_path.name == "events.csv"
        assert artifacts.jsonl_path.name == "events.jsonl"
        assert artifacts.summary_path.name == "summary.json"

    def test_debug_dir_only_exists_when_saving_crops(self, tmp_path):
        assert _artifacts(tmp_path).debug_dir is None
        with_crops = _artifacts(tmp_path, save_plate_crops=True)
        assert with_crops.debug_dir == tmp_path.resolve() / "plate_crops"
        assert with_crops.debug_dir.is_dir()


# ---------------------------------------------------------------------------
# EventLog — crossings
# ---------------------------------------------------------------------------

class TestEventLogCrossing:
    def _emit(self, tmp_path, **overrides):
        artifacts = _artifacts(tmp_path)
        log = EventLog(artifacts, STARTED_AT, collect_only=False)
        log.emit_crossing(**_crossing(**overrides))
        log.close()
        return artifacts

    def test_writes_one_row_to_each_sink(self, tmp_path):
        artifacts = self._emit(tmp_path)
        assert len(_read_jsonl(artifacts.jsonl_path)) == 1
        assert len(_read_csv(artifacts.csv_path)) == 1

    def test_csv_header_matches_the_declared_schema(self, tmp_path):
        artifacts = self._emit(tmp_path)
        with artifacts.csv_path.open(encoding="utf-8") as fh:
            assert next(csv.reader(fh)) == EVENT_CSV_FIELDS

    def test_jsonl_keeps_bbox_and_center_as_lists(self, tmp_path):
        record = _read_jsonl(self._emit(tmp_path).jsonl_path)[0]
        assert record["bbox"] == [10, 20, 110, 220]
        assert record["center"] == [60, 120]

    def test_csv_flattens_bbox_and_center_into_columns(self, tmp_path):
        row = _read_csv(self._emit(tmp_path).csv_path)[0]
        assert row["center_x"] == "60"
        assert row["center_y"] == "120"
        assert "bbox" not in row

    def test_wall_time_is_the_run_start_plus_the_media_timestamp(self, tmp_path):
        record = _read_jsonl(self._emit(tmp_path).jsonl_path)[0]
        assert record["wall_time"] == "2026-06-21T10:30:12+00:00"

    def test_plate_conf_is_rounded_for_readability(self, tmp_path):
        record = _read_jsonl(self._emit(tmp_path).jsonl_path)[0]
        assert record["plate_conf"] == 0.877

    def test_timestamp_is_rounded_to_milliseconds(self, tmp_path):
        record = _read_jsonl(self._emit(tmp_path, timestamp=1.23456).jsonl_path)[0]
        assert record["timestamp"] == 1.235

    def test_lap_fields_are_left_for_recount_to_fill(self, tmp_path):
        record = _read_jsonl(self._emit(tmp_path).jsonl_path)[0]
        assert record["lap"] == ""
        assert record["lap_time"] == ""

    def test_unresolved_crossing_is_recorded_with_an_empty_rider(self, tmp_path):
        artifacts = self._emit(
            tmp_path, rider_id="", identity_source="unresolved", plate_text="", plate_conf=0.0
        )
        record = _read_jsonl(artifacts.jsonl_path)[0]
        assert record["rider_id"] == ""
        assert record["identity_source"] == "unresolved"

    def test_jsonl_and_csv_agree_on_every_shared_field(self, tmp_path):
        artifacts = self._emit(tmp_path)
        record = _read_jsonl(artifacts.jsonl_path)[0]
        row = _read_csv(artifacts.csv_path)[0]
        for field in ("rider_id", "identity_source", "plate_text", "frame_index", "tracker_id"):
            assert row[field] == str(record[field]), field

    def test_rows_are_flushed_before_close(self, tmp_path):
        # reid-watch tails these files while detect is still running.
        artifacts = _artifacts(tmp_path)
        log = EventLog(artifacts, STARTED_AT, collect_only=False)
        log.emit_crossing(**_crossing())
        assert len(_read_jsonl(artifacts.jsonl_path)) == 1
        assert len(_read_csv(artifacts.csv_path)) == 1
        log.close()

    def test_multiple_crossings_are_appended_in_order(self, tmp_path):
        artifacts = _artifacts(tmp_path)
        log = EventLog(artifacts, STARTED_AT, collect_only=False)
        for ts in (1.0, 2.0, 3.0):
            log.emit_crossing(**_crossing(timestamp=ts))
        log.close()
        assert [r["timestamp"] for r in _read_jsonl(artifacts.jsonl_path)] == [1.0, 2.0, 3.0]

    def test_disabled_sinks_write_nothing_and_do_not_raise(self, tmp_path):
        artifacts = _artifacts(tmp_path, write_csv=False, write_jsonl=False)
        log = EventLog(artifacts, STARTED_AT, collect_only=False)
        log.emit_crossing(**_crossing())
        log.close()
        assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# EventLog — collect runs use a different schema
# ---------------------------------------------------------------------------

class TestEventLogCollect:
    def _emit(self, tmp_path):
        artifacts = _artifacts(tmp_path)
        log = EventLog(artifacts, STARTED_AT, collect_only=True)
        log.emit_collect(
            timestamp=4.5, frame_index=135, tracker_id=3,
            bbox=(1, 2, 3, 4), center=(2, 3), crop_path=tmp_path / "frame135_tid3.jpg",
        )
        log.close()
        return artifacts

    def test_csv_header_is_the_collect_schema(self, tmp_path):
        with self._emit(tmp_path).csv_path.open(encoding="utf-8") as fh:
            assert next(csv.reader(fh)) == COLLECT_CSV_FIELDS

    def test_crop_path_is_recorded(self, tmp_path):
        row = _read_csv(self._emit(tmp_path).csv_path)[0]
        assert row["crop_path"].endswith("frame135_tid3.jpg")

    def test_bbox_is_flattened_in_the_csv(self, tmp_path):
        row = _read_csv(self._emit(tmp_path).csv_path)[0]
        assert [row["bbox_x1"], row["bbox_y1"], row["bbox_x2"], row["bbox_y2"]] == ["1", "2", "3", "4"]

    def test_collect_rows_carry_no_rider_identity(self, tmp_path):
        record = _read_jsonl(self._emit(tmp_path).jsonl_path)[0]
        assert "rider_id" not in record


# ---------------------------------------------------------------------------
# CropArchive — the review surface reid-watch and humans work from
# ---------------------------------------------------------------------------

def _image(value=128):
    return np.full((40, 60, 3), value, np.uint8)


class TestCropArchive:
    def _archive(self, tmp_path):
        return CropArchive(tmp_path / "plate_crops")

    def test_disabled_archive_saves_nothing(self, tmp_path):
        assert CropArchive(None).save(1, 2, _image()) is None

    def test_disabled_archive_reports_itself_as_disabled(self, tmp_path):
        assert CropArchive(None).enabled is False
        assert self._archive(tmp_path).enabled is True

    def test_empty_crop_is_skipped(self, tmp_path):
        assert self._archive(tmp_path).save(1, 2, np.empty((0, 0, 3), np.uint8)) is None

    def test_resolved_crop_goes_into_a_per_plate_folder(self, tmp_path):
        path = self._archive(tmp_path).save(10, 4, _image(), plate_text="133")
        assert path.parent == tmp_path / "plate_crops" / "resolved" / "plate_133"

    def test_unresolved_crop_goes_into_the_unresolved_folder(self, tmp_path):
        path = self._archive(tmp_path).save(10, 4, _image(), plate_text=None)
        assert path.parent == tmp_path / "plate_crops" / "unresolved"

    def test_empty_plate_text_is_treated_as_unresolved(self, tmp_path):
        path = self._archive(tmp_path).save(10, 4, _image(), plate_text="")
        assert path.parent.name == "unresolved"

    def test_file_name_encodes_frame_and_tracker_id(self, tmp_path):
        # reid-watch parses this stem back into (frame_index, tracker_id).
        path = self._archive(tmp_path).save(10442, 1383, _image(), plate_text=None)
        assert path.name == "frame10442_tid1383.jpg"

    def test_saved_stem_round_trips_through_the_reid_watch_parser(self, tmp_path):
        from mx_tracker.reid_watch import _parse_crop_stem

        path = self._archive(tmp_path).save(10442, 1383, _image(), plate_text=None)
        assert _parse_crop_stem(path.stem) == (10442, 1383)

    def test_image_is_actually_written_and_readable(self, tmp_path):
        path = self._archive(tmp_path).save(1, 1, _image(), plate_text="7")
        assert cv2.imread(str(path)) is not None

    def test_sidecar_json_accompanies_the_crop(self, tmp_path):
        path = self._archive(tmp_path).save(1, 1, _image(), plate_text="7")
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        assert meta["frame_index"] == 1
        assert meta["tracker_id"] == 1
        assert meta["plate_text"] == "7"

    def test_unresolved_sidecar_offers_the_manual_annotation_fields(self, tmp_path):
        # A human fills manual_plate and flips save to 1; reid-watch picks it up.
        path = self._archive(tmp_path).save(1, 1, _image(), plate_text=None)
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        assert meta["manual_plate"] == ""
        assert meta["save"] == 0

    def test_resolved_sidecar_has_no_manual_fields(self, tmp_path):
        path = self._archive(tmp_path).save(1, 1, _image(), plate_text="7")
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        assert "manual_plate" not in meta
        assert "save" not in meta

    def test_sidecar_records_the_per_frame_read(self, tmp_path):
        path = self._archive(tmp_path).save(
            1, 1, _image(), plate_text=None, plate_read=PlateRead("13", 0.4321, None)
        )
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        assert meta["last_read"] == {"text": "13", "confidence": 0.432}

    def test_sidecar_records_the_vote_window_observations(self, tmp_path):
        state = TrackState()
        state.add_observation(PlateObservation("133", 0.9, 1.0, 30), ttl_sec=10.0)
        state.add_observation(PlateObservation("13", 0.4, 1.1, 33), ttl_sec=10.0)
        path = self._archive(tmp_path).save(1, 1, _image(), plate_text=None, track_state=state)
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        assert [o["text"] for o in meta["observations"]] == ["133", "13"]
        assert [o["frame_index"] for o in meta["observations"]] == [30, 33]

    def test_bike_crop_saved_alongside_an_unresolved_plate_crop(self, tmp_path):
        path = self._archive(tmp_path).save(
            5, 6, _image(), plate_text=None, bike_crop=_image(200)
        )
        assert (path.parent / "frame5_tid6_bike.jpg").exists()

    def test_bike_crop_is_not_saved_for_a_resolved_crop(self, tmp_path):
        path = self._archive(tmp_path).save(
            5, 6, _image(), plate_text="7", bike_crop=_image(200)
        )
        assert not (path.parent / "frame5_tid6_bike.jpg").exists()

    def test_empty_bike_crop_is_skipped(self, tmp_path):
        path = self._archive(tmp_path).save(
            5, 6, _image(), plate_text=None, bike_crop=np.empty((0, 0, 3), np.uint8)
        )
        assert not (path.parent / "frame5_tid6_bike.jpg").exists()

    def test_two_plates_in_one_tracker_box_land_in_separate_folders(self, tmp_path):
        archive = self._archive(tmp_path)
        first = archive.save(9, 2, _image(), plate_text="12")
        second = archive.save(9, 2, _image(), plate_text="34")
        assert first != second
        assert first.exists() and second.exists()
