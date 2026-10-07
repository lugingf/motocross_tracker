"""Running a session: detect on its stream, store what is seen, deliver it.

One session has one job. A job's identity is its run id: asking for the same
run again answers with the job that exists, and asking for a new run of the
same session replaces the old one. Everything the job sees is written to the
store before it is sent, so a restart picks up where the records stop.
"""
from __future__ import annotations

import re
import threading
import time
import urllib.parse
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

from ..config import TrackerSettings
from ..video_source import redact_source
from .delivery import Delivery, DeliveryError, HttpTransport, Rejected, RunEnded, Transport
from .engine import SCHEMA_VERSION, build_revision
from .store import ACTIVE_STATES, Job, Store

UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")

Detector = Callable[..., dict[str, Any]]
TransportFactory = Callable[[str, str], Transport]


class JobError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


def job_settings(base: TrackerSettings, configuration: dict[str, Any], data_dir: Path | None = None) -> TrackerSettings:
    """The settings a session's detection runs with: the base ones, the person's line, nothing written that is not needed."""
    calibration = configuration["calibration"]
    data = deepcopy(base.model_dump())
    a, b = calibration["line_a"], calibration["line_b"]
    data["line"]["value"] = ",".join(f"{value * 100:.4f}%" for value in (a["x"], a["y"], b["x"], b["y"]))
    data["line"]["direction"] = calibration["direction"]
    data["stream"]["rotation"] = int(calibration.get("rotation", 0))
    data["output"]["write_video"] = False
    data["output"]["write_csv"] = False
    data["output"]["write_summary"] = True
    data["output"]["write_jsonl"] = True
    data["output"]["save_plate_crops"] = True
    data["output"]["save_bike_crops_unresolved"] = True
    return TrackerSettings.model_validate(data)


def source_url(source: dict[str, Any]) -> str:
    parts = urllib.parse.urlsplit(source["url"])
    if parts.username or parts.password:
        raise JobError(400, "the source address must not carry credentials; send them as username and password")
    user = source.get("username") or ""
    if not user:
        return source["url"]
    password = source.get("password") or ""
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    userinfo = urllib.parse.quote(user, safe="") + ":" + urllib.parse.quote(password, safe="")
    return urllib.parse.urlunsplit((parts.scheme, f"{userinfo}@{host}", parts.path, parts.query, ""))


class SessionSink:
    """Writes what a run sees into the store, on the timeline of the whole session."""

    def __init__(self, store: Store, session_id: str, run_id: str, run_dir: Path, offset: float, segment: int, max_pending: int) -> None:
        self._store = store
        self._session_id = session_id
        self._run_id = run_id
        self._run_dir = run_dir
        self._offset = offset
        self._segment = segment
        self._max_pending = max_pending
        self._lock = threading.Lock()
        self.frames = 0
        self.media_ms = 0
        self.fps = 0.0
        self.backlog_ms = 0
        self.overflow = False
        self._window_started = time.monotonic()
        self._window_frames = 0
        self._segment_wall = time.monotonic()
        self._segment_ts: float | None = None
        self._touched = 0.0

    def _guard(self) -> bool:
        if self._store.pending_count(self._session_id) >= self._max_pending:
            self.overflow = True
            return False
        return True

    def crossing(self, event: dict[str, object]) -> None:
        if not self._guard():
            return
        source = str(event["identity_source"])
        identity = source if source in ("plate", "reid") else "unknown"
        x1, y1, x2, y2 = event["bbox"]  # type: ignore[misc]
        crop = str(event.get("crop_file") or "")
        evidence = ""
        if crop:
            try:
                evidence = str(Path(crop).resolve().relative_to(self._run_dir.resolve()))
            except ValueError:
                evidence = Path(crop).name
        confidence = float(event["plate_conf"]) if identity == "plate" else None  # type: ignore[arg-type]
        with self._lock:
            self._store.append_event(
                self._session_id,
                self._run_id,
                "crossing",
                round((float(event["timestamp"]) + self._offset) * 1000),  # type: ignore[arg-type]
                self._segment,
                participant_id=str(event["rider_id"]),
                plate_text=str(event["plate_text"]),
                plate_conf=confidence,
                identity_source=identity,
                bbox={"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1},
                evidence_ref=evidence,
            )

    def gap(self, segment_id: int, start: float, end: float) -> None:
        with self._lock:
            self._segment = self._store.next_segment(self._session_id)
            self._store.append_event(
                self._session_id,
                self._run_id,
                "gap",
                round((start + self._offset) * 1000),
                self._segment,
                extra={"to_ms": round((end + self._offset) * 1000), "reason": "reconnect"},
            )
            self._segment_wall = time.monotonic()
            self._segment_ts = end

    def mark_restart(self, from_ms: int, to_ms: int) -> None:
        with self._lock:
            self._store.append_event(
                self._session_id, self._run_id, "gap", from_ms, self._segment, extra={"to_ms": to_ms, "reason": "tracker_restart"}
            )

    def progress(self, frame_index: int, timestamp: float) -> None:
        now = time.monotonic()
        self.frames = frame_index
        self.media_ms = round((timestamp + self._offset) * 1000)
        if self._segment_ts is None:
            self._segment_ts = timestamp
            self._segment_wall = now
        self._window_frames += 1
        if now - self._window_started >= 5.0:
            self.fps = self._window_frames / (now - self._window_started)
            self._window_started = now
            self._window_frames = 0
        elapsed_wall = now - self._segment_wall
        elapsed_media = timestamp - self._segment_ts
        self.backlog_ms = max(0, round((elapsed_wall - elapsed_media) * 1000))
        if now - self._touched >= 2.0:
            self._touched = now
            self._store.touch(self._session_id, self.media_ms)


class Runtime:
    """One job in progress: a detection thread and the supervisor that delivers its results."""

    def __init__(self, manager: "JobManager", job: Job, source: dict[str, Any], callback: dict[str, str], detect: bool = True) -> None:
        self.manager = manager
        self._detect_enabled = detect
        self.session_id = job.session_id
        self.run_id = job.run_id
        self.source = source
        self.callback = callback
        self.stop_event = threading.Event()
        self.ended = False
        self.superseded = False
        self.error = ""
        self.sink: SessionSink | None = None
        self._wake = threading.Event()
        self._finish_asked = job.finish_requested
        self._thread = threading.Thread(target=self._supervise, name=f"job-{job.session_id[:8]}", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)

    def wake(self) -> None:
        self._wake.set()

    def finish(self) -> None:
        self._finish_asked = True
        self.manager.store.request_finish(self.session_id)
        self.stop_event.set()
        self.wake()

    def stop(self) -> None:
        self.superseded = True
        self.stop_event.set()
        self.wake()

    # -- detection --------------------------------------------------------

    def _log(self, line: str) -> None:
        self.manager.log(self.session_id, self.manager.scrub(line, self.source, self.callback))

    def _detect(self) -> None:
        manager = self.manager
        job = manager.store.get_job(self.session_id)
        run_dir = manager.run_dir(self.session_id, self.run_id)
        offset = 0.0
        segment = job.segment_id
        resumed = job.next_seq > 1 or job.last_media_ms > 0
        if resumed:
            offset = job.last_media_ms / 1000.0 + max(0.0, time.time() - job.last_wall)
            segment = manager.store.next_segment(self.session_id)
        self.sink = SessionSink(manager.store, self.session_id, self.run_id, run_dir, offset, segment, manager.settings.integration.max_outbox_events)
        if resumed:
            self.sink.mark_restart(job.last_media_ms, round(offset * 1000))
        try:
            settings = job_settings(manager.settings, job.configuration)
            manager.detector(
                source_url(self.source),
                settings,
                manager.base_dir,
                output_dir=run_dir,
                stop_event=self.stop_event,
                logger=self._log,
                event_sink=self.sink,
            )
        except Exception as exc:
            self.error = manager.scrub(str(exc), self.source, self.callback) or exc.__class__.__name__

    # -- supervision ------------------------------------------------------

    def _supervise(self) -> None:
        manager = self.manager
        store = manager.store
        settings = manager.settings.integration
        transport = manager.transport_factory(self.callback["base_url"], self.callback["token"])
        delivery = Delivery(store, transport, self.session_id, self.run_id, settings.batch_size, manager.run_dir_for(self.session_id))
        store.set_state(self.session_id, "running")

        detector = threading.Thread(target=self._detect if self._detect_enabled else (lambda: None), name=f"detect-{self.session_id[:8]}", daemon=True)
        detector.start()

        last_heartbeat = 0.0
        last_revision = time.monotonic()
        last_revised_seq = 0
        backoff = manager.retry_delay
        try:
            while True:
                alive = detector.is_alive()
                try:
                    if not delivery.flush_events() or not delivery.flush_evidence():
                        self.superseded = True
                        self.stop_event.set()
                    now = time.monotonic()
                    if now - last_heartbeat >= settings.heartbeat_interval_sec or not alive:
                        ack = delivery.heartbeat(
                            "finishing" if self._finish_asked else "running",
                            self.sink.media_ms if self.sink else 0,
                            self.sink.frames if self.sink else 0,
                            self.sink.fps if self.sink else 0.0,
                            self.sink.backlog_ms if self.sink else 0,
                        )
                        last_heartbeat = now
                        if ack.get("stop"):
                            self.superseded = True
                            self.stop_event.set()
                        elif ack.get("finish") and not self._finish_asked:
                            self.finish()
                    if alive and not self.superseded:
                        due = manager.consume_revision_request(self.session_id)
                        job = store.get_job(self.session_id)
                        if (due or (now - last_revision >= settings.revision_interval_sec and job.acked_seq > last_revised_seq)) and job.acked_seq > 0:
                            manager.submit_revision(delivery, self.session_id, self.run_id, "provisional")
                            last_revised_seq = job.acked_seq
                            last_revision = now
                    backoff = manager.retry_delay
                except RunEnded as ended:
                    self._log(f"run ended by lap_vision: {ended}")
                    self.superseded = True
                    self.stop_event.set()
                except Rejected as rejected:
                    self._log(str(rejected))
                except DeliveryError as failure:
                    self._log(str(failure))
                    self._wake.wait(backoff)
                    backoff = min(backoff * 2, 30.0)
                if self.sink is not None and self.sink.overflow:
                    self.error = "the outbox is full; lap_vision is not taking events"
                    self.stop_event.set()
                if not alive:
                    break
                self._wake.wait(manager.poll_interval)
                self._wake.clear()
        finally:
            detector.join()

        self._finalize(delivery)

    def _finalize(self, delivery: Delivery) -> None:
        manager = self.manager
        store = manager.store
        if self.superseded:
            store.set_state(self.session_id, "stopped")
            self.ended = True
            return
        if self.error:
            # A request to finish does not make a failure a success. The detection stopped for a
            # reason; saying "finished" would hand over a result that is missing whatever it missed.
            store.set_state(self.session_id, "failed", self.error)
            try:
                delivery.heartbeat("failed", self.sink.media_ms if self.sink else 0, self.sink.frames if self.sink else 0, 0.0, 0, self.error)
            except (DeliveryError, RunEnded, Rejected):
                pass
            self.ended = True
            return

        store.set_state(self.session_id, "finishing")
        backoff = manager.retry_delay
        while not manager.closing.is_set():
            try:
                if not delivery.flush_events() or not delivery.flush_evidence(force=True):
                    store.set_state(self.session_id, "stopped")
                    break
                job = store.get_job(self.session_id)
                if job.acked_seq == store.last_seq(self.session_id):
                    manager.submit_revision(delivery, self.session_id, self.run_id, "final")
                    store.set_state(self.session_id, "finished")
                    break
            except RunEnded:
                store.set_state(self.session_id, "stopped")
                break
            except (DeliveryError, Rejected) as failure:
                self._log(str(failure))
            self._wake.wait(backoff)
            backoff = min(backoff * 2, 30.0)
        self.ended = True


class JobManager:
    def __init__(
        self,
        settings: TrackerSettings,
        base_dir: Path,
        detector: Detector | None = None,
        transport_factory: TransportFactory | None = None,
        logger: Callable[[str], None] | None = None,
    ) -> None:
        self.settings = settings
        self.base_dir = base_dir
        if detector is None:
            from ..pipeline import run_stream_detection

            detector = run_stream_detection
        self.detector = detector
        timeout = settings.integration.request_timeout_sec
        self.transport_factory = transport_factory or (lambda base, token: HttpTransport(base, token, timeout))
        self._logger = logger or (lambda line: print(line, flush=True))
        state_dir = Path(settings.integration.state_dir).expanduser()
        if not state_dir.is_absolute():
            state_dir = (base_dir / state_dir).resolve()
        self.state_dir = state_dir
        self.store = Store(state_dir / "state.sqlite")
        self.closing = threading.Event()
        self._submit_lock = threading.Lock()
        self.poll_interval = 0.5
        self.retry_delay = 0.5
        self._lock = threading.RLock()
        self._runtimes: dict[str, Runtime] = {}
        self._callbacks: dict[str, dict[str, str]] = {}
        self._revision_requests: set[str] = set()
        for job in self.store.list_jobs(ACTIVE_STATES):
            self.store.set_state(job.session_id, "interrupted")

    # -- plumbing ---------------------------------------------------------

    def log(self, session_id: str, line: str) -> None:
        self._logger(f"[integration {session_id[:8]}] {line}")

    @staticmethod
    def scrub(text: str, source: dict[str, Any], callback: dict[str, str]) -> str:
        for secret in (source.get("password"), source.get("username"), callback.get("token")):
            if secret:
                text = text.replace(str(secret), "***")
        return text.replace(source.get("url", "\0"), redact_source(source.get("url", "")))

    def run_dir(self, session_id: str, run_id: str) -> Path:
        return self.state_dir / "runs" / session_id / run_id

    def run_dir_for(self, session_id: str) -> Callable[[str], Path]:
        return lambda run_id: self.run_dir(session_id, run_id)

    def consume_revision_request(self, session_id: str) -> bool:
        with self._lock:
            if session_id in self._revision_requests:
                self._revision_requests.discard(session_id)
                return True
        return False

    def active(self) -> int:
        with self._lock:
            return sum(1 for runtime in self._runtimes.values() if not runtime.ended)

    def close(self) -> None:
        self.closing.set()
        with self._lock:
            runtimes = list(self._runtimes.values())
        for runtime in runtimes:
            runtime.stop_event.set()
            runtime.wake()
        for runtime in runtimes:
            runtime.join(5)
        self.store.close()

    # -- revisions --------------------------------------------------------

    def send_pending_revisions(self, delivery: Delivery, session_id: str) -> None:
        """Send every revision that was prepared and not acknowledged, oldest first.

        A revision is written down before it is sent and forgotten only once lap_vision has
        answered, so an answer that was lost leads to the same revision being sent again under the
        same id, which lap_vision recognises, and not to a second revision with a new one. One that
        lap_vision refuses and that is only provisional has been overtaken by events and is dropped.
        """
        for submission in self.store.pending_submissions(session_id):
            try:
                delivery.revision(submission)
            except Rejected:
                if submission["status"] != "final":
                    self.store.drop_submission(session_id, submission["submission_id"])
                    continue
                raise
            self.store.drop_submission(session_id, submission["submission_id"])

    def submit_revision(self, delivery: Delivery, session_id: str, run_id: str, status: str) -> None:
        with self._submit_lock:
            self.send_pending_revisions(delivery, session_id)
            job = self.store.get_job(session_id)
            events = self.store.events_upto(session_id, job.acked_seq)
            if status == "final":
                # The same events, corrections and settings are the same final revision. Sending it
                # again after an answer was lost must not make a second one.
                version = int(job.configuration.get("version", 0))
                submission_id = f"{run_id[:8]}-final-{job.acked_seq}-{job.corrections_up_to}-{version}"
            else:
                submission_id = f"{run_id[:8]}-r{self.store.next_revision(session_id)}"
            submission = build_revision(
                events,
                job.corrections,
                job.configuration,
                job.race_start_ms,
                self.settings.integration.low_confidence,
                run_id,
                submission_id,
                status,
                job.corrections_up_to,
            )
            self.store.save_submission(session_id, submission)
            self.send_pending_revisions(delivery, session_id)

    # -- requests ---------------------------------------------------------

    def status(self, session_id: str) -> dict[str, Any]:
        job = self.store.get_job(session_id)
        if job is None:
            raise JobError(404, "no such job")
        return {
            "schema_version": SCHEMA_VERSION,
            "job_id": job.run_id,
            "session_id": job.session_id,
            "run_id": job.run_id,
            "state": job.state,
            "last_seq": job.acked_seq,
            "outbox_pending": self.store.pending_count(session_id),
            "error": job.error,
        }

    def find(self, job_id: str) -> Job:
        for job in self.store.list_jobs():
            if job.run_id == job_id:
                return job
        raise JobError(404, "no such job")

    def validate(self, request: dict[str, Any]) -> None:
        integration = self.settings.integration
        if request.get("schema_version") != SCHEMA_VERSION:
            raise JobError(400, "unsupported schema_version")
        for key in ("session_id", "run_id"):
            if not UUID.match(str(request.get(key, ""))):
                raise JobError(400, f"{key} must be a uuid")
        source = request.get("source") or {}
        url = str(source.get("url", ""))
        if not any(url.startswith(prefix) for prefix in integration.allowed_sources):
            raise JobError(403, "this source is not allowed")
        callback = request.get("callback") or {}
        if not any(str(callback.get("base_url", "")).startswith(prefix) for prefix in integration.allowed_callbacks):
            raise JobError(403, "this callback address is not allowed")
        if not callback.get("token"):
            raise JobError(400, "callback.token is required")
        calibration = (request.get("configuration") or {}).get("calibration")
        if not calibration:
            raise JobError(400, "the configuration has no calibration")
        try:
            job_settings(self.settings, request["configuration"])
        except Exception as exc:
            raise JobError(400, f"the calibration cannot be used: {exc}") from None

    def create(self, request: dict[str, Any]) -> dict[str, Any]:
        self.validate(request)
        session_id, run_id = request["session_id"], request["run_id"]
        configuration = request["configuration"]
        race_start = request.get("race_start_media_ms")
        callback = {"base_url": request["callback"]["base_url"], "token": request["callback"]["token"]}

        with self._lock:
            job = self.store.get_job(session_id)
            runtime = self._runtimes.get(session_id)
            if job is not None and job.run_id == run_id:
                if runtime is not None and not runtime.ended:
                    self._callbacks[session_id] = callback
                    return self.status(session_id)
                if job.state in ("finished", "stopped"):
                    self._callbacks[session_id] = callback
                    return self.status(session_id)
            if self.active() >= self.settings.integration.max_jobs and not (runtime and not runtime.ended):
                raise JobError(503, "the tracker is busy with another session")
            if runtime is not None and not runtime.ended:
                runtime.stop()
                runtime.join(10)

            if job is None:
                job = self.store.create_job(session_id, run_id, configuration, race_start)
            else:
                if job.run_id != run_id:
                    self.store.set_run(session_id, run_id, "queued")
                else:
                    self.store.set_state(session_id, "queued")
                self.store.set_configuration(session_id, configuration, race_start)
                job = self.store.get_job(session_id)
            self._callbacks[session_id] = callback
            runtime = Runtime(self, job, request["source"], callback, detect=not job.finish_requested)
            self._runtimes[session_id] = runtime
        runtime.start()
        return self.status(session_id)

    def finish(self, job_id: str, request: dict[str, Any] | None = None) -> dict[str, Any]:
        job = self.find(job_id)
        with self._lock:
            runtime = self._runtimes.get(job.session_id)
            if runtime is not None and not runtime.ended:
                runtime.finish()
                return self.status(job.session_id)
            if job.state in ("finished", "stopped"):
                return self.status(job.session_id)

            callback = (request or {}).get("callback") or {}
            if not any(str(callback.get("base_url", "")).startswith(prefix) for prefix in self.settings.integration.allowed_callbacks):
                raise JobError(403, "this callback address is not allowed")
            if not callback.get("token"):
                raise JobError(400, "callback.token is required")
            if self.active() >= self.settings.integration.max_jobs:
                raise JobError(503, "the tracker is busy with another session")

            # The input is closed already, or the job is not running: whatever is stored is all there
            # will be, so it is finished from the store without reading the stream.
            self.store.request_finish(job.session_id)
            self.store.set_state(job.session_id, "queued")
            self._callbacks[job.session_id] = {"base_url": callback["base_url"], "token": callback["token"]}
            runtime = Runtime(self, self.store.get_job(job.session_id), {}, self._callbacks[job.session_id], detect=False)
            self._runtimes[job.session_id] = runtime
        runtime.start()
        return self.status(job.session_id)

    def stop(self, job_id: str) -> dict[str, Any]:
        job = self.find(job_id)
        with self._lock:
            runtime = self._runtimes.get(job.session_id)
        if runtime is not None and not runtime.ended:
            runtime.stop()
            runtime.join(10)
        else:
            self.store.set_state(job.session_id, "stopped")
        return self.status(job.session_id)

    def recompute(self, job_id: str, request: dict[str, Any]) -> dict[str, Any]:
        job = self.find(job_id)
        callback = request.get("callback") or {}
        if not any(str(callback.get("base_url", "")).startswith(prefix) for prefix in self.settings.integration.allowed_callbacks):
            raise JobError(403, "this callback address is not allowed")
        if not callback.get("token"):
            raise JobError(400, "callback.token is required")
        configuration = request.get("configuration") or {}
        calibration = configuration.get("calibration") or job.configuration.get("calibration")
        merged = dict(configuration)
        merged["calibration"] = calibration
        self.store.set_configuration(job.session_id, merged, request.get("race_start_media_ms"))
        self.store.set_corrections(job.session_id, request.get("corrections") or [], int(request.get("up_to_correction_id") or 0))

        with self._lock:
            runtime = self._runtimes.get(job.session_id)
            self._callbacks[job.session_id] = {"base_url": callback["base_url"], "token": callback["token"]}
            if runtime is not None and not runtime.ended:
                self._revision_requests.add(job.session_id)
                runtime.wake()
                return self.status(job.session_id)
        threading.Thread(target=self._recompute_idle, args=(job.session_id, job.run_id), daemon=True).start()
        return self.status(job.session_id)

    def _recompute_idle(self, session_id: str, run_id: str) -> None:
        callback = self._callbacks[session_id]
        delivery = Delivery(self.store, self.transport_factory(callback["base_url"], callback["token"]), session_id, run_id, self.settings.integration.batch_size, self.run_dir_for(session_id))
        backoff = self.retry_delay
        for _ in range(8):
            try:
                if not delivery.flush_events():
                    return
                job = self.store.get_job(session_id)
                if job.acked_seq != self.store.last_seq(session_id):
                    continue
                status = "final" if job.state == "finished" else "provisional"
                self.submit_revision(delivery, session_id, run_id, status)
                return
            except RunEnded:
                return
            except (DeliveryError, Rejected) as failure:
                self.log(session_id, str(failure))
            time.sleep(backoff)
            backoff = min(backoff * 2, 10.0)

    def health(self) -> dict[str, Any]:
        active = self.active()
        return {
            "schema_version": SCHEMA_VERSION,
            "state": "busy" if active >= self.settings.integration.max_jobs else "ready",
            "active_jobs": active,
            "version": "1",
        }
