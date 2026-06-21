from __future__ import annotations

import json
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .config import load_settings
from .pipeline import collect_samples, run_file_detection, run_stream_detection
from .training import build_dataset, train_model, validate_dataset


@dataclass
class JobRecord:
    job_id: str
    action: str
    payload: dict[str, Any]
    status: str = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    logs: list[str] = field(default_factory=list)
    stop_event: threading.Event = field(default_factory=threading.Event, repr=False)
    thread: threading.Thread | None = field(default=None, repr=False)

    def log(self, message: str) -> None:
        self.logs.append(message)
        self.logs[:] = self.logs[-200:]

    def public(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "action": self.action,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "result": self.result,
            "error": self.error,
            "logs": self.logs[-50:],
            "payload": self.payload,
        }


class JobManager:
    def __init__(self, default_config_path: str | Path | None = None) -> None:
        self.default_config_path = None if default_config_path is None else str(Path(default_config_path).expanduser().resolve())
        self.jobs: dict[str, JobRecord] = {}
        self.lock = threading.Lock()

    def list_jobs(self) -> list[dict[str, Any]]:
        with self.lock:
            return [job.public() for job in self.jobs.values()]

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self.lock:
            job = self.jobs.get(job_id)
            return None if job is None else job.public()

    def stop_job(self, job_id: str) -> bool:
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                return False
            job.stop_event.set()
            job.log("stop requested")
            return True

    def submit(self, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        job = JobRecord(job_id=uuid.uuid4().hex[:12], action=action, payload=payload)
        thread = threading.Thread(target=self._run, args=(job,), daemon=True)
        job.thread = thread
        with self.lock:
            self.jobs[job.job_id] = job
        thread.start()
        return job.public()

    def _run(self, job: JobRecord) -> None:
        job.status = "running"
        job.started_at = time.time()
        try:
            config_path = job.payload.get("config_path") or self.default_config_path
            settings, base_dir = load_settings(config_path)
            action = job.action
            if action == "detect_file":
                source = str(job.payload["source"])
                result = run_file_detection(
                    source=source,
                    settings=settings,
                    base_dir=base_dir,
                    output_dir=job.payload.get("output_dir"),
                    limit_frames=job.payload.get("limit_frames"),
                    calibrate_line=bool(job.payload.get("calibrate_line", False)),
                    logger=job.log,
                )
            elif action == "detect_stream":
                source = str(job.payload["source"])
                result = run_stream_detection(
                    source=source,
                    settings=settings,
                    base_dir=base_dir,
                    output_dir=job.payload.get("output_dir"),
                    stop_event=job.stop_event,
                    limit_frames=job.payload.get("limit_frames"),
                    calibrate_line=bool(job.payload.get("calibrate_line", False)),
                    logger=job.log,
                )
            elif action == "collect":
                source = str(job.payload["source"])
                source_mode = str(job.payload.get("source_mode", "file"))
                result = collect_samples(
                    source=source,
                    mode=source_mode,
                    settings=settings,
                    base_dir=base_dir,
                    output_dir=job.payload.get("output_dir"),
                    stop_event=job.stop_event,
                    limit_frames=job.payload.get("limit_frames"),
                    calibrate_line=bool(job.payload.get("calibrate_line", False)),
                    logger=job.log,
                )
            elif action == "dataset_build":
                result = build_dataset(
                    raw_dir=job.payload["raw_dir"],
                    dataset_dir=job.payload["dataset_dir"],
                    train_ratio=float(job.payload.get("train_ratio", 0.8)),
                    seed=int(job.payload.get("seed", 0)),
                    clean=bool(job.payload.get("clean", False)),
                    include_unlabeled=bool(job.payload.get("include_unlabeled", False)),
                    classes=job.payload.get("classes"),
                ).to_dict()
            elif action == "dataset_validate":
                result = validate_dataset(job.payload["dataset_dir"]).to_dict()
            elif action == "train":
                result = train_model(
                    data_yaml=job.payload["data_yaml"],
                    model_path=job.payload.get("model_path", "data/models/yolov8n.pt"),
                    project_dir=job.payload.get("project_dir", "data/runs/detect"),
                    run_name=job.payload.get("run_name", "train"),
                    epochs=int(job.payload.get("epochs", 100)),
                    imgsz=int(job.payload.get("imgsz", 640)),
                    batch=int(job.payload.get("batch", 16)),
                    device=str(job.payload.get("device", "auto")),
                    workers=int(job.payload.get("workers", 8)),
                )
            else:
                raise ValueError(f"Unsupported action: {action}")
            stoppable = {"detect_stream", "collect"}
            job.status = "stopped" if job.stop_event.is_set() and action in stoppable else "completed"
            job.result = result
        except Exception as exc:
            job.status = "failed"
            job.error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            job.log("".join(traceback.format_exception(exc)))
        finally:
            job.finished_at = time.time()


class JsonHandler(BaseHTTPRequestHandler):
    manager: JobManager

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b"{}"
        payload = json.loads(body.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object")
        return payload

    def _send(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send(HTTPStatus.OK, {"status": "ok"})
            return
        if self.path == "/jobs":
            self._send(HTTPStatus.OK, {"jobs": self.manager.list_jobs()})
            return
        if self.path.startswith("/jobs/"):
            job_id = self.path.split("/")[2]
            job = self.manager.get_job(job_id)
            if job is None:
                self._send(HTTPStatus.NOT_FOUND, {"error": "job not found"})
                return
            self._send(HTTPStatus.OK, job)
            return
        self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:
        try:
            if self.path == "/jobs":
                payload = self._read_json()
                action = str(payload.get("action", "")).strip()
                if not action:
                    self._send(HTTPStatus.BAD_REQUEST, {"error": "action is required"})
                    return
                job = self.manager.submit(action, payload)
                self._send(HTTPStatus.ACCEPTED, job)
                return
            if self.path.startswith("/jobs/") and self.path.endswith("/stop"):
                job_id = self.path.split("/")[2]
                if not self.manager.stop_job(job_id):
                    self._send(HTTPStatus.NOT_FOUND, {"error": "job not found"})
                    return
                self._send(HTTPStatus.OK, {"job_id": job_id, "status": "stop_requested"})
                return
            self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})
        except Exception as exc:
            self._send(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def log_message(self, _format: str, *_args: object) -> None:
        return


def run_service(
    host: str,
    port: int,
    default_config_path: str | Path | None = None,
) -> None:
    manager = JobManager(default_config_path=default_config_path)

    class Handler(JsonHandler):
        pass

    Handler.manager = manager
    server = ThreadingHTTPServer((host, port), Handler)
    try:
        print(f"listening on http://{host}:{port}")
        server.serve_forever()
    except KeyboardInterrupt:
        print("service stopped")
    finally:
        server.server_close()
