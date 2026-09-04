"""End-to-end tests for SourceProcessor with fake models and a scripted source.

These drive a whole run — frames in, events.jsonl/events.csv out — while
substituting the two heavy dependencies (`VideoSource` and `DetectionModels`).
That covers the orchestration the pure-logic tests can't reach: the crossing
gate, the cooldown, plate voting across frames, and the crop routing for
resolved vs unresolved crossings.

The scripted source lets a test say "this track's centre walks this path" and
assert on the events that fall out.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from mx_tracker import pipeline as pipeline_module
from mx_tracker.config import TrackerSettings
from mx_tracker.pipeline import PlateRead, SourceProcessor
from mx_tracker.video_source import FramePacket


FRAME_W, FRAME_H = 400, 200
FPS = 10.0
# Vertical finish line down the middle of the frame.
LINE_VALUE = "50%,0%,50%,100%"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _Tensor:
    def __init__(self, array):
        self._array = np.asarray(array, dtype=float)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._array


class _Boxes:
    def __init__(self, detections):
        # detections: list of (tracker_id, x1, y1, x2, y2, class_id)
        self._n = len(detections)
        self.xyxy = _Tensor([[d[1], d[2], d[3], d[4]] for d in detections])
        self.cls = _Tensor([d[5] for d in detections])
        self.id = _Tensor([d[0] for d in detections]) if detections else None

    def __len__(self):
        return self._n


class _Result:
    def __init__(self, detections):
        self.boxes = _Boxes(detections) if detections else None


class FakePlateReader:
    """Returns scripted plate readings, recording how often it was asked."""

    def __init__(self, best=None, groups=None):
        self._best = best if best is not None else PlateRead(None, 0.0, None)
        self._groups = groups if groups is not None else []
        self.read_best_calls = 0
        self.read_all_calls = 0

    @property
    def available(self):
        return True

    def read_best(self, crop):
        self.read_best_calls += 1
        return self._best

    def read_all(self, crop):
        self.read_all_calls += 1
        return list(self._groups)


class FakeReid:
    def __init__(self, answer=None):
        self.answer = answer
        self.calls = 0

    def identify(self, crop):
        self.calls += 1
        return self.answer


class FakeModels:
    def __init__(self, frames_detections, plate_reader=None, reid=None):
        self._frames = frames_detections
        self.plate_reader = plate_reader or FakePlateReader()
        self.reid = reid
        self.device = "cpu"

    def track(self, frame, settings):
        # frame_index is 1-based; the scripted list is 0-based.
        index = self._call_count
        self._call_count += 1
        detections = self._frames[index] if index < len(self._frames) else []
        return _Result(detections)

    _call_count = 0


class FakeVideoSource:
    """Yields one FramePacket per scripted frame, at a fixed fps."""

    def __init__(self, frame_count):
        self.frame_count = frame_count

    def probe(self):
        return {"fps": FPS, "width": FRAME_W, "height": FRAME_H, "reconnects": 0}

    def frames(self, stop_event=None):
        for index in range(1, self.frame_count + 1):
            yield FramePacket(
                frame=np.zeros((FRAME_H, FRAME_W, 3), np.uint8),
                frame_index=index,
                timestamp=(index - 1) / FPS,
                fps=FPS,
                width=FRAME_W,
                height=FRAME_H,
            )


def _bbox_at(centre_x, centre_y=100, half=30):
    return (centre_x - half, centre_y - half, centre_x + half, centre_y + half)


def _path(tracker_id, centres):
    """One detection per frame, walking the track's centre along `centres`."""
    return [[(tracker_id, *_bbox_at(x), 3)] for x in centres]


@pytest.fixture
def run(monkeypatch, tmp_path):
    """Run a scripted source through SourceProcessor and return (summary, run_dir)."""

    def _run(frames_detections, *, plate_reader=None, reid=None, settings=None, **kwargs):
        settings = settings or _settings()
        models = FakeModels(frames_detections, plate_reader=plate_reader, reid=reid)
        models._call_count = 0
        monkeypatch.setattr(
            pipeline_module, "VideoSource",
            lambda source, mode, s: FakeVideoSource(len(frames_detections)),
        )
        monkeypatch.setattr(
            pipeline_module.DetectionModels, "load",
            classmethod(lambda cls, s, base_dir, collect_only, logger: models),
        )
        out_dir = tmp_path / kwargs.pop("out_name", "run")
        processor = SourceProcessor(
            source="scripted.mp4", mode="file", settings=settings,
            base_dir=tmp_path, output_dir=out_dir, logger=lambda m: None, **kwargs,
        )
        return processor.run(), out_dir, models

    return _run


def _settings(**line_overrides) -> TrackerSettings:
    settings = TrackerSettings()
    settings.line.value = LINE_VALUE
    settings.line.width = 20
    settings.line.direction = "left_to_right"
    settings.line.cooldown_sec = 2.0
    settings.output.write_video = False
    settings.output.write_summary = False
    settings.models.min_bike_crop_px = 1
    # These tests exercise the crossing logic on synthetic black frames, which
    # carry no motion for the gate to see. TestMotionGate covers the gate.
    settings.motion_gate.enabled = False
    for key, value in line_overrides.items():
        setattr(settings.line, key, value)
    return settings


def _events(run_dir: Path) -> list[dict]:
    path = run_dir / "events.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# The crossing gate
# ---------------------------------------------------------------------------

class TestCrossingGate:
    def test_bike_crossing_left_to_right_is_counted_once(self, run):
        reader = FakePlateReader(groups=[PlateRead("133", 0.9, None, x_center=50.0)])
        summary, run_dir, _ = run(_path(1, [40, 120, 180, 260, 340]), plate_reader=reader)
        assert summary["events"] == 1
        assert summary["crossing_counts"] == {"plate_133": 1}

    def test_bike_crossing_the_wrong_way_is_not_counted(self, run):
        reader = FakePlateReader(groups=[PlateRead("133", 0.9, None, x_center=50.0)])
        summary, _, _ = run(_path(1, [340, 260, 180, 120, 40]), plate_reader=reader)
        assert summary["events"] == 0
        assert summary["unresolved_crossings"] == 0

    def test_bike_that_never_reaches_the_line_is_not_counted(self, run):
        reader = FakePlateReader(groups=[PlateRead("133", 0.9, None, x_center=50.0)])
        summary, _, _ = run(_path(1, [20, 40, 60, 80, 100]), plate_reader=reader)
        assert summary["events"] == 0

    def test_bike_that_enters_the_band_and_reverses_is_not_counted(self, run):
        """Regression: band entry/exit used to register as a left_to_right lap.

        Nothing here ever crosses the line, and the crossings are far enough
        apart in time that the cooldown cannot mask a phantom event.
        """
        reader = FakePlateReader(groups=[PlateRead("133", 0.9, None, x_center=50.0)])
        # The line sits at x=200 with a 20 px band (190..210). Every centre
        # here stays right of 200, so the track enters the band, touches 202
        # and reverses without ever changing sides.
        centres = [260, 230, 212, 205, 202, 205, 212, 230, 260, 300]
        assert all(x > 200 for x in centres)
        # Repeat each step so the whole manoeuvre spans several cooldown
        # windows — a phantom crossing would have room to be emitted.
        long_path = [c for c in centres for _ in range(6)]
        summary, run_dir, _ = run(_path(1, long_path), plate_reader=reader)
        assert summary["events"] == 0, _events(run_dir)
        assert summary["unresolved_crossings"] == 0

    def test_two_passes_separated_by_more_than_the_cooldown_count_twice(self, run):
        reader = FakePlateReader(groups=[PlateRead("133", 0.9, None, x_center=50.0)])
        # Cross, drive back out of frame slowly, then cross again 3 s later.
        first = [40, 120, 180, 260, 340]
        wait = [340] * 30          # 3 s of standing still, right of the line
        back = [340, 260, 180, 120, 40]   # returns right→left (not counted)
        second = [40, 120, 180, 260, 340]
        summary, _, _ = run(_path(1, first + wait + back + second), plate_reader=reader)
        assert summary["crossing_counts"] == {"plate_133": 2}

    def test_second_crossing_inside_the_cooldown_is_dropped(self, run):
        reader = FakePlateReader(groups=[PlateRead("133", 0.9, None, x_center=50.0)])
        # Two full crossings within 1 s — the cooldown keeps only the first.
        path = [40, 180, 340, 180, 40, 180, 340]
        summary, _, _ = run(_path(1, path), plate_reader=reader)
        assert summary["crossing_counts"] == {"plate_133": 1}

    def test_two_tracks_are_counted_independently(self, run):
        reader = FakePlateReader(groups=[PlateRead("7", 0.9, None, x_center=50.0)])
        frames = [
            [(1, *_bbox_at(x), 3), (2, *_bbox_at(x - 10), 3)]
            for x in (40, 120, 180, 260, 340)
        ]
        summary, run_dir, _ = run(frames, plate_reader=reader)
        assert summary["events"] == 2
        assert {e["tracker_id"] for e in _events(run_dir)} == {1, 2}

    def test_non_motorcycle_classes_are_ignored(self, run):
        reader = FakePlateReader(groups=[PlateRead("133", 0.9, None, x_center=50.0)])
        car = [[(1, *_bbox_at(x), 2)] for x in (40, 120, 180, 260, 340)]
        summary, _, _ = run(car, plate_reader=reader)
        assert summary["events"] == 0

    def test_frames_without_track_ids_are_skipped(self, run):
        summary, _, _ = run([[], [], []])
        assert summary["events"] == 0

    def test_limit_frames_stops_the_run_early(self, run):
        reader = FakePlateReader(groups=[PlateRead("133", 0.9, None, x_center=50.0)])
        # The crossing happens on frame 4; stopping at 2 must miss it.
        summary, _, _ = run(_path(1, [40, 120, 180, 260, 340]), plate_reader=reader, limit_frames=2)
        assert summary["events"] == 0


# ---------------------------------------------------------------------------
# Identity resolution at the crossing
# ---------------------------------------------------------------------------

class TestIdentityResolution:
    def _cross(self):
        return _path(1, [40, 120, 180, 260, 340])

    def test_plate_read_at_the_crossing_names_the_rider(self, run):
        reader = FakePlateReader(groups=[PlateRead("44", 0.8, None, x_center=50.0)])
        _, run_dir, _ = run(self._cross(), plate_reader=reader)
        event = _events(run_dir)[0]
        assert event["identity_source"] == "plate"
        assert event["plate_text"] == "44"
        assert event["rider_id"] == "plate_44"

    def test_voted_plate_is_used_when_the_crossing_frame_reads_nothing(self, run):
        # read_all finds nothing at the crossing, but earlier per-frame scans did.
        reader = FakePlateReader(best=PlateRead("91", 0.7, None, x_center=10.0), groups=[])
        _, run_dir, _ = run(self._cross(), plate_reader=reader)
        event = _events(run_dir)[0]
        assert event["rider_id"] == "plate_91"
        assert event["identity_source"] == "plate"

    def test_crossing_with_no_plate_anywhere_is_unresolved(self, run):
        summary, run_dir, _ = run(self._cross(), plate_reader=FakePlateReader())
        assert summary["unresolved_crossings"] == 1
        assert summary["events"] == 0
        event = _events(run_dir)[0]
        assert event["identity_source"] == "unresolved"
        assert event["rider_id"] == ""

    def test_reid_resolves_a_crossing_with_no_plate(self, run):
        reid = FakeReid(answer="plate_26")
        summary, run_dir, _ = run(self._cross(), plate_reader=FakePlateReader(), reid=reid)
        assert reid.calls == 1
        assert summary["unresolved_crossings"] == 0
        assert _events(run_dir)[0]["identity_source"] == "reid"

    def test_reid_is_not_consulted_when_a_plate_was_read(self, run):
        reid = FakeReid(answer="plate_26")
        reader = FakePlateReader(groups=[PlateRead("44", 0.8, None, x_center=50.0)])
        run(self._cross(), plate_reader=reader, reid=reid)
        assert reid.calls == 0

    def test_two_bikes_in_one_tracker_box_emit_two_events(self, run):
        # Two digit groups in one crop: both riders get their own event.
        reader = FakePlateReader(groups=[
            PlateRead("12", 0.8, None, x_center=20.0),
            PlateRead("34", 0.8, None, x_center=200.0),
        ])
        summary, run_dir, _ = run(self._cross(), plate_reader=reader)
        assert summary["events"] == 2
        assert summary["crossing_counts"] == {"plate_12": 1, "plate_34": 1}

    def test_the_second_of_two_bikes_is_offset_in_time(self, run):
        reader = FakePlateReader(groups=[
            PlateRead("12", 0.8, None, x_center=20.0),
            PlateRead("34", 0.8, None, x_center=200.0),
        ])
        _, run_dir, _ = run(self._cross(), plate_reader=reader)
        stamps = [e["timestamp"] for e in _events(run_dir)]
        assert stamps[1] - stamps[0] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Per-frame plate scanning
# ---------------------------------------------------------------------------

class TestPlateScanning:
    def test_scanning_only_happens_near_the_line(self, run):
        # read_distance_multiplier=2.5 × width 20 → scan within 50 px.
        reader = FakePlateReader()
        far_away = _path(1, [20, 25, 30, 35, 40])
        run(far_away, plate_reader=reader)
        assert reader.read_best_calls == 0

    def test_scanning_happens_when_the_track_is_near_the_line(self, run):
        reader = FakePlateReader()
        run(_path(1, [180, 185, 190, 195]), plate_reader=reader)
        assert reader.read_best_calls > 0

    def test_scan_every_n_frames_throttles_the_plate_model(self, run):
        settings = _settings()
        settings.reads.scan_every_n_frames = 3
        reader = FakePlateReader()
        centres = [190] * 9
        run(_path(1, centres), plate_reader=reader, settings=settings)
        # Frames 3, 6 and 9 of nine are scanned.
        assert reader.read_best_calls == 3

    def test_collect_only_never_calls_the_plate_reader(self, run):
        reader = FakePlateReader()
        run(_path(1, [40, 120, 180, 260, 340]), plate_reader=reader, collect_only=True)
        assert reader.read_best_calls == 0
        assert reader.read_all_calls == 0


# ---------------------------------------------------------------------------
# Run outputs
# ---------------------------------------------------------------------------

class TestRunOutputs:
    def _cross(self):
        return _path(1, [40, 120, 180, 260, 340])

    def test_run_info_records_the_source(self, run):
        _, run_dir, _ = run(self._cross())
        info = json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))
        assert info["source"] == "scripted.mp4"
        assert "started_at" in info

    def test_summary_reports_the_resolved_line(self, run):
        summary, _, _ = run(self._cross())
        assert summary["line"] == LINE_VALUE
        assert summary["mode"] == "file"

    def test_summary_carries_the_probed_source_stats(self, run):
        summary, _, _ = run(self._cross())
        assert summary["source_stats"]["fps"] == pytest.approx(FPS)

    def test_events_csv_and_jsonl_hold_the_same_number_of_rows(self, run):
        reader = FakePlateReader(groups=[PlateRead("133", 0.9, None, x_center=50.0)])
        _, run_dir, _ = run(self._cross(), plate_reader=reader)
        csv_rows = (run_dir / "events.csv").read_text(encoding="utf-8").strip().splitlines()
        assert len(csv_rows) - 1 == len(_events(run_dir))

    def test_summary_json_is_written_when_enabled(self, run):
        settings = _settings()
        settings.output.write_summary = True
        _, run_dir, _ = run(self._cross(), settings=settings)
        assert json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))["mode"] == "file"

    def test_unresolved_crop_is_archived_for_review(self, run):
        settings = _settings()
        settings.output.save_plate_crops = True
        summary, run_dir, _ = run(self._cross(), settings=settings)
        assert summary["saved_crops"] == 1
        assert list((run_dir / "plate_crops" / "unresolved").glob("*.jpg"))

    def test_resolved_crop_is_filed_under_its_plate(self, run):
        settings = _settings()
        settings.output.save_plate_crops = True
        reader = FakePlateReader(groups=[PlateRead("133", 0.9, None, x_center=50.0)])
        _, run_dir, _ = run(self._cross(), plate_reader=reader, settings=settings)
        assert (run_dir / "plate_crops" / "resolved" / "plate_133").is_dir()

    def test_no_crops_are_saved_when_disabled(self, run):
        _, run_dir, _ = run(self._cross())
        assert not (run_dir / "plate_crops").exists()

    def test_recount_can_read_the_events_this_run_produced(self, run):
        from mx_tracker.recount import recount

        reader = FakePlateReader(groups=[PlateRead("133", 0.9, None, x_center=50.0)])
        _, run_dir, _ = run(self._cross(), plate_reader=reader)
        results = recount(run_dir, logger=lambda m: None)
        assert results.exists()
        assert "plate_133" in results.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Collect mode
# ---------------------------------------------------------------------------

class TestCollectMode:
    def test_crossing_saves_a_bike_crop_instead_of_an_event(self, run):
        summary, run_dir, _ = run(_path(1, [40, 120, 180, 260, 340]), collect_only=True)
        assert summary["saved_crops"] == 1
        assert summary["events"] == 0
        assert list(run_dir.glob("frame*_tid1.jpg"))

    def test_collect_rows_name_the_saved_crop(self, run):
        _, run_dir, _ = run(_path(1, [40, 120, 180, 260, 340]), collect_only=True)
        record = _events(run_dir)[0]
        assert Path(record["crop_path"]).exists()


# ---------------------------------------------------------------------------
# Stale track eviction
# ---------------------------------------------------------------------------

class TestTrackEviction:
    def test_track_state_is_dropped_after_its_ttl(self, run):
        settings = _settings()
        settings.reads.track_state_ttl_sec = 0.2
        reader = FakePlateReader(groups=[PlateRead("133", 0.9, None, x_center=50.0)])
        # Track 1 appears, disappears for well over the TTL, then reappears.
        frames = _path(1, [40, 120]) + [[] for _ in range(20)] + _path(1, [180, 260, 340])
        summary, _, _ = run(frames, plate_reader=reader, settings=settings)
        # The gap wiped last_center, so the reappearance starts a fresh track
        # and still gets counted when it crosses.
        assert summary["events"] == 1

    def test_live_tracks_are_not_evicted(self, run):
        settings = _settings()
        settings.reads.track_state_ttl_sec = 10.0
        reader = FakePlateReader(groups=[PlateRead("133", 0.9, None, x_center=50.0)])
        summary, _, _ = run(_path(1, [40, 120, 180, 260, 340]), plate_reader=reader, settings=settings)
        assert summary["events"] == 1



# ---------------------------------------------------------------------------
# The motion gate, from the pipeline's side
#
# The gate must never cost an event, so these drive real motion through the
# source: a bright blob walks across the frame while the fake tracker reports
# the matching box, so the gate sees the same thing a detector would.
#
# The fake tracker is keyed on the frame index the pipeline is actually on
# (not on how many times it has been called) — gated frames never reach it,
# exactly as with the real model.
# ---------------------------------------------------------------------------

ENTRY_ZONE_X = FRAME_W // 4       # gate watches x < 100 for left_to_right
LINE_X = FRAME_W // 2             # the finish line sits at x = 200


BLOB = 40  # must be smaller than ENTRY_ZONE_X so the blob can clear the zone


def _blob_frame(x, size=BLOB, value=255):
    frame = np.zeros((FRAME_H, FRAME_W, 3), np.uint8)
    if x is None:
        return frame
    x1, x2 = max(0, x - size // 2), min(FRAME_W, x + size // 2)
    y1, y2 = (FRAME_H - size) // 2, (FRAME_H + size) // 2
    if x2 > x1:
        frame[y1:y2, x1:x2] = value
    return frame


class _BlobSource:
    """Frames carrying a blob at the scripted centres, announcing each index."""

    def __init__(self, centres, cursor):
        self.centres = centres
        self.cursor = cursor

    def probe(self):
        return {"fps": FPS, "width": FRAME_W, "height": FRAME_H, "reconnects": 0}

    def frames(self, stop_event=None):
        for index, x in enumerate(self.centres, start=1):
            self.cursor["index"] = index
            yield FramePacket(
                frame=_blob_frame(x), frame_index=index, timestamp=(index - 1) / FPS,
                fps=FPS, width=FRAME_W, height=FRAME_H,
            )


class _IndexedModels:
    """Reports the detection scripted for whichever frame is being processed."""

    def __init__(self, centres, cursor, plate_reader, reid=None):
        self.centres = centres
        self.cursor = cursor
        self.plate_reader = plate_reader
        self.reid = reid
        self.device = "cpu"
        self.track_calls = 0

    def track(self, frame, settings):
        self.track_calls += 1
        x = self.centres[self.cursor["index"] - 1]
        return _Result([] if x is None else [(1, *_bbox_at(x, half=BLOB // 2), 3)])


@pytest.fixture
def run_gated(monkeypatch, tmp_path):
    """Run a blob path through SourceProcessor with the gate live."""

    def _run(centres, *, plate=None, gate=None, out_name="gated"):
        settings = _settings()
        settings.motion_gate.enabled = True
        settings.motion_gate.warmup_frames = 2
        settings.motion_gate.hold_sec = 0.5
        settings.motion_gate.min_area_fraction = 0.002
        for key, value in (gate or {}).items():
            setattr(settings.motion_gate, key, value)

        reader = FakePlateReader(groups=[PlateRead(plate, 0.9, None, x_center=50.0)]) if plate else FakePlateReader()
        cursor = {"index": 1}
        models = _IndexedModels(centres, cursor, reader)
        monkeypatch.setattr(pipeline_module, "VideoSource",
                            lambda source, mode, s: _BlobSource(centres, cursor))
        monkeypatch.setattr(pipeline_module.DetectionModels, "load",
                            classmethod(lambda cls, s, b, c, l: models))
        out_dir = tmp_path / out_name
        processor = SourceProcessor(
            source="blob.mp4", mode="file", settings=settings, base_dir=tmp_path,
            output_dir=out_dir, logger=lambda m: None,
        )
        return processor.run(), models

    return _run


# A settled, empty scene before the rider arrives. Without it the background
# is seeded from a frame that already contains the rider, and everything after
# reads as motion — the same way a real run needs its warmup on empty track.
IDLE_LEAD = [None] * 20


def _enters_and_crosses():
    """A rider entering through the entry zone and crossing the line at x=200."""
    return IDLE_LEAD + list(range(20, 400, 20))


def _appears_past_the_entry_zone():
    """A rider whose whole extent stays clear of the entry zone."""
    return IDLE_LEAD + list(range(ENTRY_ZONE_X + BLOB, 400, 20))


class TestMotionGate:
    def test_gate_and_its_zone_are_reported(self, run_gated):
        summary, _ = run_gated(_enters_and_crosses())
        assert summary["motion_gate"]["enabled"] is True
        assert summary["motion_gate"]["zone"] == [0, 0, ENTRY_ZONE_X, FRAME_H]

    def test_a_rider_entering_and_crossing_is_counted(self, run_gated):
        summary, _ = run_gated(_enters_and_crosses(), plate="133")
        assert summary["crossing_counts"] == {"plate_133": 1}

    def test_the_hold_carries_the_rider_from_the_entry_zone_to_the_line(self, run_gated):
        """Entry-zone motion only wakes detection; the hold sustains it.

        By the time the rider reaches the line at x=200 it has long left the
        entry zone, so only `note_detection` keeps detection running.
        """
        summary, models = run_gated(_enters_and_crosses(), plate="133")
        assert summary["crossing_counts"] == {"plate_133": 1}
        # Detection ran well past the frames whose blob touched the zone.
        in_zone = [x for x in _enters_and_crosses() if x is not None and x - BLOB // 2 < ENTRY_ZONE_X]
        assert models.track_calls > len(in_zone)

    def test_idle_frames_before_the_rider_are_skipped(self, run_gated):
        summary, _ = run_gated(_enters_and_crosses(), plate="133")
        assert summary["motion_gate"]["gated_frames"] > 0
        assert summary["crossing_counts"] == {"plate_133": 1}

    def test_the_tracker_is_not_called_on_gated_frames(self, run_gated):
        summary, models = run_gated(_enters_and_crosses(), plate="133")
        stats = summary["motion_gate"]
        assert models.track_calls == stats["processed_frames"]
        assert models.track_calls < stats["processed_frames"] + stats["gated_frames"]

    def test_gating_changes_nothing_about_the_events(self, run_gated):
        centres = _enters_and_crosses() + [None] * 40
        gated, _ = run_gated(centres, plate="133", out_name="with_gate")
        plain, _ = run_gated(centres, plate="133", gate={"enabled": False}, out_name="no_gate")
        assert gated["events"] == plain["events"]
        assert gated["crossing_counts"] == plain["crossing_counts"]
        assert gated["unresolved_crossings"] == plain["unresolved_crossings"]
        assert gated["motion_gate"]["gated_frames"] > 0
        assert plain["motion_gate"]["gated_frames"] == 0

    def test_a_static_source_gates_nearly_every_frame(self, run_gated):
        summary, models = run_gated([None] * 60)
        assert summary["motion_gate"]["gated_fraction"] > 0.9
        assert summary["events"] == 0

    def test_disabled_gate_processes_every_frame(self, run_gated):
        summary, models = run_gated([None] * 30, gate={"enabled": False})
        assert summary["motion_gate"]["gated_frames"] == 0
        assert models.track_calls == 30

    def test_a_rider_appearing_past_the_entry_zone_is_missed_while_gated(self, run_gated):
        """The gate's one real assumption, made explicit.

        Riders are expected to travel in through the entry zone. One that
        materialises beyond it — revealed by an occlusion clearing, or
        entering the frame past the strip — produces no motion where the gate
        looks and no detection to hold it open, so it goes uncounted.
        """
        summary, _ = run_gated(_appears_past_the_entry_zone(), plate="133")
        assert summary["events"] == 0
        assert summary["motion_gate"]["gated_fraction"] > 0.9

    def test_widening_the_zone_recovers_a_rider_that_skips_the_entry_strip(self, run_gated):
        # zone_fraction=1.0 watches the whole frame: maximum safety, less saving.
        summary, _ = run_gated(
            _appears_past_the_entry_zone(), plate="133", gate={"zone_fraction": 1.0},
        )
        assert summary["crossing_counts"] == {"plate_133": 1}
        assert summary["motion_gate"]["gated_frames"] > 0

    def test_without_a_hold_the_rider_is_dropped_on_leaving_the_entry_zone(self, run_gated):
        """Why hold_sec exists, and why it must not be set to zero.

        With no hold, detection stops the frame after the rider clears the
        entry zone — long before it reaches the line.
        """
        summary, _ = run_gated(_enters_and_crosses(), plate="133", gate={"hold_sec": 0.0})
        assert summary["events"] == 0

    def test_the_default_hold_is_long_enough_to_reach_the_line(self, run_gated):
        summary, _ = run_gated(_enters_and_crosses(), plate="133")
        assert summary["crossing_counts"] == {"plate_133": 1}
