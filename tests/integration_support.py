import json
import threading
import time
from pathlib import Path

from mx_tracker.config import TrackerSettings
from mx_tracker.integration.delivery import DeliveryError, RunEnded, Rejected
from mx_tracker.integration.jobs import JobManager

JPEG = bytes([0xFF, 0xD8, 0xFF, 0xE0, 0, 16]) + b"JFIF" + bytes(8) + bytes([0xFF, 0xD9])

FIXTURES = Path(__file__).parent / "fixtures" / "contract"
SESSION = "5d0f6c9e-6a3c-4c35-9a3e-0b3f0f3a0001"
RUN = "8f4a1d52-2f0b-4a53-8a7c-0b3f0f3a0002"
OTHER_RUN = "9a4a1d52-2f0b-4a53-8a7c-0b3f0f3a0003"


def fixture(name):
    return json.loads((FIXTURES / name).read_text())


def settings_for(tmp_path, **overrides):
    integration = {
        "enabled": True,
        "token": "tracker-token",
        "state_dir": str(tmp_path / "state"),
        "allowed_sources": ["rtsp://127.0.0.1:18554/"],
        "allowed_callbacks": ["https://lapvision.org/api"],
        "heartbeat_interval_sec": 0.05,
        "revision_interval_sec": 0.2,
    }
    integration.update(overrides)
    return TrackerSettings.model_validate({"integration": integration})


def job_request(**changes):
    request = fixture("job_request.json")
    request["source"]["url"] = "rtsp://127.0.0.1:18554/imports/5d0f6c9e"
    request["source"]["username"] = "r-5d0f6c9e"
    request["source"]["password"] = "read-secret"
    request.update(changes)
    return request


class FakeTransport:
    """Stands in for lap_vision: acknowledges what it is sent, and can be told to fail."""

    def __init__(self):
        self.posts = []
        self.fail = None
        self.stop_after = None
        self.finish = False
        self.reject_evidence = False
        self.lock = threading.Lock()
        self.acked = 0

    def post(self, path, payload):
        with self.lock:
            if self.fail is not None:
                raise self.fail
            self.posts.append((path, payload))
            if path.endswith("/events"):
                events = payload["events"]
                if events:
                    expected = self.acked + 1
                    for event in events:
                        if event["seq"] == expected:
                            expected += 1
                    self.acked = expected - 1
                stop = self.stop_after is not None and self.acked >= self.stop_after
                return {"acked_seq": self.acked, "accepted": len(events), "stop": stop}
            if path.endswith("/evidence"):
                if self.reject_evidence:
                    raise Rejected(409, "no such event")
                return {"stored": len(payload["items"]), "stop": False}
            if path.endswith("/heartbeat"):
                return {"stop": False, "finish": self.finish}
            if path.endswith("/revisions"):
                return {"revision": sum(1 for p, _ in self.posts if p.endswith("/revisions")), "stop": False}
            raise AssertionError(path)

    def paths(self, suffix):
        with self.lock:
            return [payload for path, payload in self.posts if path.endswith(suffix)]


class FakeDetector:
    """Plays a script of crossings into the sink, then waits to be stopped."""

    def __init__(self, crossings=(), fail=None, hold=True, fail_on_stop=None, pictures=False):
        self.pictures = pictures
        self.crossings = list(crossings)
        self.fail = fail
        self.fail_on_stop = fail_on_stop
        self.hold = hold
        self.started = threading.Event()
        self.calls = []

    def __call__(self, source, settings, base_dir, output_dir=None, stop_event=None, logger=None, event_sink=None):
        self.calls.append({"source": source, "settings": settings, "output_dir": output_dir})
        self.started.set()
        if self.fail:
            raise RuntimeError(self.fail)
        for index, (ts, rider) in enumerate(self.crossings, start=1):
            crop = ""
            if self.pictures:
                folder = Path(output_dir) / "plate_crops" / ("resolved" if rider.startswith("plate_") else "unresolved")
                folder.mkdir(parents=True, exist_ok=True)
                crop = str(folder / f"frame{index}_tid1.jpg")
                Path(crop).write_bytes(JPEG)
                if not rider.startswith("plate_"):
                    (folder / f"frame{index}_tid1_bike.jpg").write_bytes(JPEG)
            event_sink.progress(index, ts)
            event_sink.crossing(
                {
                    "timestamp": ts,
                    "rider_id": rider,
                    "identity_source": "plate" if rider.startswith("plate_") else "unresolved",
                    "plate_text": rider.removeprefix("plate_") if rider.startswith("plate_") else "",
                    "plate_conf": 0.9,
                    "bbox": [10, 20, 110, 220],
                    "crop_file": crop,
                }
            )
        if self.hold:
            stop_event.wait(10)
        if self.fail_on_stop:
            raise RuntimeError(self.fail_on_stop)
        return {}


def make_manager(tmp_path, detector, transport, **overrides):
    manager = JobManager(
        settings_for(tmp_path, **overrides),
        tmp_path,
        detector=detector,
        transport_factory=lambda base, token: transport,
        logger=lambda line: None,
    )
    manager.poll_interval = 0.02
    manager.retry_delay = 0.02
    return manager


def wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False
