import time
from threading import Event

import cv2
import numpy as np
import pytest

from mx_tracker.config import TrackerSettings
from mx_tracker.video_source import VideoSource, redact_source


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    for index in range(10):
        writer.write(np.full((48, 64, 3), index * 20, dtype=np.uint8))
    writer.release()
    return str(path)


def settings(**stream):
    base = TrackerSettings()
    for key, value in stream.items():
        setattr(base.stream, key, value)
    return base


def pyav_source(path, **stream):
    source = VideoSource(path, "stream", settings(**stream))
    source.stats = source._probe_pyav()
    return source


def test_timestamps_follow_the_media_clock_when_the_consumer_is_slow(clip):
    source = pyav_source(clip, max_reconnects=0)

    stamps = []
    for packet in source._frames_pyav(Event()):
        stamps.append(packet.timestamp)
        time.sleep(0.05)

    assert len(stamps) == 10
    assert stamps == pytest.approx([i / 10.0 for i in range(10)], abs=0.02)


def test_a_reconnect_starts_a_segment_and_reports_the_gap(clip):
    source = pyav_source(clip, max_reconnects=1, reconnect_delay_sec=0.3)

    packets = list(source._frames_pyav(Event()))

    assert [p.segment_id for p in packets] == [0] * 10 + [1] * 10
    assert packets[0].gap_before == 0.0
    assert packets[10].gap_before >= 0.3
    stamps = [p.timestamp for p in packets]
    assert stamps == sorted(stamps)
    assert packets[10].timestamp - packets[9].timestamp >= 0.3
    assert source.stats["reconnects"] == 2


def test_stop_ends_the_stream_between_frames(clip):
    source = pyav_source(clip, max_reconnects=5)
    stop = Event()

    seen = 0
    for _ in source._frames_pyav(stop):
        seen += 1
        if seen == 3:
            stop.set()

    assert seen == 3


def test_a_source_that_cannot_be_opened_fails_loudly_and_without_its_credentials():
    source = VideoSource("rtsp://user:secret@127.0.0.1:1/imports/x", "stream", settings(open_timeout_sec=0.5))

    with pytest.raises(RuntimeError) as raised:
        source.probe()

    assert "secret" not in str(raised.value)
    assert "127.0.0.1:1/imports/x" in str(raised.value)


def test_redact_source_drops_credentials_and_query():
    assert redact_source("rtsp://user:pw@host:8554/imports/a?token=1") == "rtsp://host:8554/imports/a"
    assert redact_source("/data/video/clip.mp4") == "/data/video/clip.mp4"


def test_rotation_turns_the_frames_and_reports_the_turned_size(clip):
    source = VideoSource(clip, "file", settings(rotation=90))

    packet = next(iter(source.frames(Event())))

    assert packet.frame.shape[:2] == (64, 48)
    assert (packet.width, packet.height) == (48, 64)


def test_an_unsupported_rotation_is_refused():
    with pytest.raises(ValueError):
        TrackerSettings.model_validate({"stream": {"rotation": 45}})
