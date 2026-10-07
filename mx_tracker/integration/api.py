"""The HTTP surface lap_vision calls, on a listener of its own.

Every route needs the bearer token. Nothing else the tracker serves is reachable
here, and nothing here accepts a path or a directory: a source must start with
one of the configured prefixes.
"""
from __future__ import annotations

import hmac
import json
import os
import re
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import cv2

from ..config import TrackerSettings
from ..video_source import redact_source
from .jobs import JobError, JobManager, source_url

MAX_BODY = 4 << 20
PREFIX = "/integration/v1"
JOB_PATH = re.compile(r"^/integration/v1/jobs/([0-9a-f-]{36})(/finish|/stop|/recompute)?$")


def configured_token(settings: TrackerSettings) -> str:
    return settings.integration.token or os.environ.get("MX_INTEGRATION_TOKEN", "")


def grab_frame(source: dict[str, Any], settings: TrackerSettings) -> bytes:
    """One frame of a stream as a JPEG, in the orientation the camera sends it."""
    import av

    stream = settings.stream
    options = {"rtsp_transport": stream.rtsp_transport} if source["url"].startswith("rtsp") else {}
    container = av.open(source_url(source), options=options, timeout=(stream.open_timeout_sec, stream.read_timeout_sec))
    try:
        for frame in container.decode(container.streams.video[0]):
            ok, encoded = cv2.imencode(".jpg", frame.to_ndarray(format="bgr24"), [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                raise RuntimeError("the frame could not be encoded")
            return encoded.tobytes()
    finally:
        container.close()
    raise RuntimeError("the stream has no frames")


class IntegrationHandler(BaseHTTPRequestHandler):
    manager: JobManager
    token: str
    server_version = "mx-tracker-integration"

    def _send(self, status: int, payload: Any, content_type: str = "application/json") -> None:
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        header = self.headers.get("Authorization", "")
        supplied = header[7:].strip() if header.lower().startswith("bearer ") else ""
        return bool(self.token) and hmac.compare_digest(supplied.encode(), self.token.encode())

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            raise JobError(413, "the request is too large")
        body = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            raise JobError(400, "the body is not JSON") from None
        if not isinstance(payload, dict):
            raise JobError(400, "the body must be a JSON object")
        return payload

    def _route(self, method: str) -> None:
        if not self._authorized():
            self._send(HTTPStatus.UNAUTHORIZED, {"error": {"message": "unauthorized"}})
            return
        path = urllib.parse.urlsplit(self.path).path
        try:
            if method == "GET" and path == f"{PREFIX}/health":
                self._send(HTTPStatus.OK, self.manager.health())
                return
            if method == "POST" and path == f"{PREFIX}/jobs":
                self._send(HTTPStatus.ACCEPTED, self.manager.create(self._read_json()))
                return
            if method == "POST" and path == f"{PREFIX}/preview":
                self._preview(self._read_json())
                return
            match = JOB_PATH.match(path)
            if match:
                job_id, action = match.group(1), match.group(2)
                if method == "GET" and action is None:
                    self._send(HTTPStatus.OK, self.manager.status(self.manager.find(job_id).session_id))
                    return
                if method == "POST" and action == "/finish":
                    self._send(HTTPStatus.ACCEPTED, self.manager.finish(job_id, self._read_json()))
                    return
                if method == "POST" and action == "/stop":
                    self._send(HTTPStatus.OK, self.manager.stop(job_id))
                    return
                if method == "POST" and action == "/recompute":
                    self._send(HTTPStatus.ACCEPTED, self.manager.recompute(job_id, self._read_json()))
                    return
            self._send(HTTPStatus.NOT_FOUND, {"error": {"message": "not found"}})
        except JobError as error:
            self._send(error.status, {"error": {"message": str(error)}})
        except Exception as error:
            self.manager.log("-", f"request failed: {error.__class__.__name__}")
            self._send(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": {"message": "internal error"}})

    def _preview(self, payload: dict[str, Any]) -> None:
        source = payload.get("source") or {}
        url = str(source.get("url", ""))
        if not any(url.startswith(prefix) for prefix in self.manager.settings.integration.allowed_sources):
            raise JobError(403, "this source is not allowed")
        try:
            image = grab_frame(source, self.manager.settings)
        except Exception as error:
            message = JobManager.scrub(str(error), source, {})
            raise JobError(502, f"no frame could be taken from {redact_source(url)}: {message}") from None
        self._send(HTTPStatus.OK, image, "image/jpeg")

    def do_GET(self) -> None:
        self._route("GET")

    def do_POST(self) -> None:
        self._route("POST")

    def log_message(self, _format: str, *_args: object) -> None:
        return


def make_server(settings: TrackerSettings, manager: JobManager) -> ThreadingHTTPServer:
    token = configured_token(settings)
    if not token:
        raise RuntimeError("integration.token or MX_INTEGRATION_TOKEN is required")

    class Handler(IntegrationHandler):
        pass

    Handler.manager = manager
    Handler.token = token
    return ThreadingHTTPServer((settings.integration.host, settings.integration.port), Handler)
