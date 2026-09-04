"""Tests for video_source.py — frame indexing, timestamps and looping.

Timestamps are the clock everything downstream runs on: the crossing cooldown,
the vote window and track TTL eviction all subtract two `FramePacket`
timestamps. They must therefore start at zero, advance by one frame duration,
and never move backwards — including across a looped file.

A tiny real MP4 is written with cv2 so the OpenCV backend behaviour (rather
than a mock of it) is what gets exercised.
"""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from mx_tracker.config import TrackerSettings
from mx_tracker.video_source import (
    FramePacket,
    VideoSource,
    _is_local_file_source,
    _is_numeric_source,
    _open_capture,
)


FPS = 10.0
WIDTH, HEIGHT = 64, 48
FRAMES = 5


@pytest.fixture
def clip(tmp_path):
    """A 5-frame, 10 fps MP4 — one frame every 0.1 s."""
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    assert writer.isOpened(), "cv2 cannot write mp4 in this environment"
    for index in range(FRAMES):
        writer.write(np.full((HEIGHT, WIDTH, 3), index * 40, np.uint8))
    writer.release()
    assert path.stat().st_size > 0
    return path


def _settings(**stream_overrides) -> TrackerSettings:
    settings = TrackerSettings()
    for key, value in stream_overrides.items():
        setattr(settings.stream, key, value)
    return settings


def _take(iterator, count):
    packets = []
    for packet in iterator:
        packets.append(packet)
        if len(packets) >= count:
            break
    return packets


# ---------------------------------------------------------------------------
# Source kind detection
# ---------------------------------------------------------------------------

class TestSourceKind:
    def test_digit_string_is_a_camera_index(self):
        assert _is_numeric_source("0") is True

    def test_path_is_not_numeric(self, clip):
        assert _is_numeric_source(str(clip)) is False

    def test_existing_file_is_local(self, clip):
        assert _is_local_file_source(str(clip)) is True

    def test_camera_index_is_not_a_local_file(self):
        assert _is_local_file_source("0") is False

    def test_stream_url_is_not_a_local_file(self):
        assert _is_local_file_source("rtsp://camera/stream") is False

    def test_missing_path_is_not_a_local_file(self, tmp_path):
        assert _is_local_file_source(str(tmp_path / "nope.mp4")) is False

    def test_unopenable_source_raises_with_the_source_in_the_message(self, tmp_path):
        missing = str(tmp_path / "nope.mp4")
        with pytest.raises(RuntimeError, match="nope.mp4"):
            _open_capture(missing)


# ---------------------------------------------------------------------------
# probe()
# ---------------------------------------------------------------------------

class TestProbe:
    def test_reports_clip_geometry_and_fps(self, clip):
        stats = VideoSource(str(clip), "file", _settings()).probe()
        assert (stats["width"], stats["height"]) == (WIDTH, HEIGHT)
        assert stats["fps"] == pytest.approx(FPS)

    def test_reconnect_counter_starts_at_zero(self, clip):
        assert VideoSource(str(clip), "file", _settings()).probe()["reconnects"] == 0

    def test_unusable_fps_falls_back_to_the_configured_value(self, clip, monkeypatch):
        # Some backends report 0 or NaN instead of a frame rate.
        settings = _settings()
        settings.runtime.source_fps_fallback = 25.0
        real_get = cv2.VideoCapture.get

        def fake_get(self, prop):
            return float("nan") if prop == cv2.CAP_PROP_FPS else real_get(self, prop)

        monkeypatch.setattr(cv2.VideoCapture, "get", fake_get)
        assert VideoSource(str(clip), "file", settings).probe()["fps"] == pytest.approx(25.0)

    def test_zero_fps_falls_back_to_the_configured_value(self, clip, monkeypatch):
        settings = _settings()
        settings.runtime.source_fps_fallback = 25.0
        real_get = cv2.VideoCapture.get
        monkeypatch.setattr(
            cv2.VideoCapture,
            "get",
            lambda self, prop: 0.0 if prop == cv2.CAP_PROP_FPS else real_get(self, prop),
        )
        assert VideoSource(str(clip), "file", settings).probe()["fps"] == pytest.approx(25.0)

    def test_frames_probes_implicitly_when_not_probed_yet(self, clip):
        source = VideoSource(str(clip), "file", _settings())
        assert source.stats == {}
        next(source.frames())
        assert source.stats["width"] == WIDTH


# ---------------------------------------------------------------------------
# File mode
# ---------------------------------------------------------------------------

class TestFileMode:
    def test_yields_every_frame_once(self, clip):
        packets = list(VideoSource(str(clip), "file", _settings()).frames())
        assert len(packets) == FRAMES

    def test_frame_index_is_one_based_and_contiguous(self, clip):
        packets = list(VideoSource(str(clip), "file", _settings()).frames())
        assert [p.frame_index for p in packets] == list(range(1, FRAMES + 1))

    def test_first_frame_starts_at_zero(self, clip):
        first = next(VideoSource(str(clip), "file", _settings()).frames())
        assert first.timestamp == pytest.approx(0.0)

    def test_timestamps_advance_by_one_frame_duration(self, clip):
        packets = list(VideoSource(str(clip), "file", _settings()).frames())
        expected = [index / FPS for index in range(FRAMES)]
        assert [p.timestamp for p in packets] == pytest.approx(expected, abs=1e-3)

    def test_no_two_frames_share_a_timestamp(self, clip):
        # Regression: the first frame's genuine 0.0 was read as "unavailable"
        # and replaced by 1/fps, colliding with the second frame.
        stamps = [p.timestamp for p in VideoSource(str(clip), "file", _settings()).frames()]
        assert len(set(stamps)) == len(stamps)

    def test_packet_carries_frame_geometry_and_fps(self, clip):
        first = next(VideoSource(str(clip), "file", _settings()).frames())
        assert isinstance(first, FramePacket)
        assert (first.width, first.height) == (WIDTH, HEIGHT)
        assert first.fps == pytest.approx(FPS)
        assert first.frame.shape == (HEIGHT, WIDTH, 3)

    def test_falls_back_to_frame_index_when_the_backend_has_no_position(self, clip, monkeypatch):
        real_get = cv2.VideoCapture.get
        monkeypatch.setattr(
            cv2.VideoCapture,
            "get",
            lambda self, prop: 0.0 if prop == cv2.CAP_PROP_POS_MSEC else real_get(self, prop),
        )
        stamps = [p.timestamp for p in VideoSource(str(clip), "file", _settings()).frames()]
        assert stamps == pytest.approx([index / FPS for index in range(FRAMES)], abs=1e-3)

    def test_stop_event_ends_iteration(self, clip):
        from threading import Event

        stop = Event()
        stop.set()
        assert list(VideoSource(str(clip), "file", _settings()).frames(stop)) == []


# ---------------------------------------------------------------------------
# Looping a file
#
# loop_file replays a finite file forever. The media clock rewinds on each
# pass, but the emitted timestamps must not: a backwards jump makes the
# crossing cooldown suppress real laps and stops stale tracks being evicted.
# ---------------------------------------------------------------------------

class TestLoopedFile:
    def test_keeps_yielding_past_the_end_of_the_file(self, clip):
        packets = _take(VideoSource(str(clip), "file", _settings(loop_file=True)).frames(), FRAMES + 3)
        assert len(packets) == FRAMES + 3

    def test_frame_index_keeps_growing_across_passes(self, clip):
        packets = _take(VideoSource(str(clip), "file", _settings(loop_file=True)).frames(), FRAMES + 3)
        assert [p.frame_index for p in packets] == list(range(1, FRAMES + 4))

    def test_timestamps_never_move_backwards(self, clip):
        stamps = [
            p.timestamp
            for p in _take(VideoSource(str(clip), "file", _settings(loop_file=True)).frames(), FRAMES * 2 + 1)
        ]
        assert all(later > earlier for earlier, later in zip(stamps, stamps[1:])), stamps

    def test_second_pass_continues_from_the_first(self, clip):
        stamps = [
            p.timestamp
            for p in _take(VideoSource(str(clip), "file", _settings(loop_file=True)).frames(), FRAMES + 1)
        ]
        # The pass boundary is one frame duration wide, like every other step.
        assert stamps[FRAMES] - stamps[FRAMES - 1] == pytest.approx(1 / FPS, abs=1e-3)


# ---------------------------------------------------------------------------
# Stream mode over a local file (the `--loop-source` replay path)
# ---------------------------------------------------------------------------

class TestStreamModeOverLocalFile:
    def test_timestamps_are_wall_clock_and_monotonic(self, clip):
        packets = list(VideoSource(str(clip), "stream", _settings(max_reconnects=0)).frames())
        stamps = [p.timestamp for p in packets]
        assert stamps == sorted(stamps)
        assert stamps[0] >= 0.0

    def test_max_reconnects_zero_stops_after_the_first_pass(self, clip):
        packets = list(VideoSource(str(clip), "stream", _settings(max_reconnects=0, reconnect_delay_sec=0.0)).frames())
        assert len(packets) == FRAMES

    def test_reconnect_count_is_reported_in_stats(self, clip):
        source = VideoSource(str(clip), "stream", _settings(max_reconnects=1, reconnect_delay_sec=0.0))
        list(source.frames())
        assert source.stats["reconnects"] == 2
