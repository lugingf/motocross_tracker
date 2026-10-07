"""Sending what the store holds to lap_vision.

Nothing here decides anything about laps. It posts events in order, waits for
the acknowledgement before forgetting that they are pending, and stops the
whole job when lap_vision says that the run is no longer wanted.
"""
from __future__ import annotations

import base64
import json
import logging
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Protocol

from .engine import SCHEMA_VERSION
from .store import Store


class DeliveryError(Exception):
    """The delivery failed in a way that a later attempt may cure."""


class RunEnded(Exception):
    """lap_vision no longer wants this run: it was closed, replaced or the token is no longer valid."""


class Rejected(Exception):
    """lap_vision refused the content. Sending it again will not change the answer."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"lap_vision answered {status}: {message}")
        self.status = status


class Transport(Protocol):
    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]: ...


class HttpTransport:
    def __init__(self, base_url: str, token: str, timeout: float) -> None:
        self._base = base_url.rstrip("/")
        self._token = token
        self._timeout = timeout

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            self._base + path,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                body = response.read()
        except urllib.error.HTTPError as error:
            message = _message(error.read())
            if error.code in (401, 410):
                raise RunEnded(f"lap_vision answered {error.code}: {message}") from None
            if error.code in (400, 409, 413, 422):
                raise Rejected(error.code, message) from None
            raise DeliveryError(f"lap_vision answered {error.code}: {message}") from None
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise DeliveryError(f"lap_vision is not reachable: {getattr(error, 'reason', error)}") from None
        return json.loads(body) if body else {}


def _message(body: bytes) -> str:
    try:
        return str(json.loads(body)["error"]["message"])
    except Exception:
        return body.decode("utf-8", "replace")[:200]


MAX_PICTURE_BYTES = 480 << 10
PICTURES_PER_BATCH = 5

log = logging.getLogger(__name__)


def fit_picture(data: bytes, limit: int = MAX_PICTURE_BYTES) -> bytes | None:
    """The picture as a JPEG that is no larger than `limit`, scaled down if it has to be; None if it is not an image."""
    if len(data) <= limit:
        return data
    import cv2
    import numpy as np

    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return None
    scale = 0.8
    while scale > 0.1:
        small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok and len(encoded) <= limit:
            return encoded.tobytes()
        scale *= 0.8
    return None


class Delivery:
    def __init__(
        self,
        store: Store,
        transport: Transport,
        session_id: str,
        run_id: str,
        batch_size: int,
        run_dir: Callable[[str], Path] | None = None,
    ) -> None:
        self._run_dir = run_dir
        self._evidence_failures = 0
        self._evidence_retry_at = 0.0
        self._store = store
        self._transport = transport
        self._session_id = session_id
        self._run_id = run_id
        self._batch_size = batch_size
        self._prefix = f"/internal/video-imports/{session_id}"

    def flush_events(self) -> bool:
        """Send every pending event. Answers whether the session wants the work to go on."""
        while True:
            job = self._store.get_job(self._session_id)
            batch = self._store.events_after(self._session_id, job.acked_seq, self._batch_size)
            if not batch:
                return True
            events = [_wire(event) for event in batch]
            ack = self._transport.post(
                f"{self._prefix}/events",
                {"schema_version": SCHEMA_VERSION, "run_id": self._run_id, "events": events},
            )
            self._store.acknowledge(self._session_id, int(ack["acked_seq"]))
            if ack.get("stop"):
                return False
            if int(ack["acked_seq"]) < batch[-1]["seq"]:
                return True

    def _pictures_of(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        """The picture files of one crossing, as the contract wants them: the plate, and the rider when no number was read."""
        if self._run_dir is None:
            return []
        reference = Path(event["evidence_ref"])
        base = self._run_dir(event["run_id"])
        candidates = [("plate", base / reference)]
        if not event["plate_text"]:
            candidates.append(("bike", base / reference.with_name(reference.stem + "_bike" + reference.suffix)))
        items = []
        for kind, path in candidates:
            try:
                data = fit_picture(path.read_bytes())
            except OSError:
                continue
            if data:
                items.append(
                    {"event_id": event["event_id"], "kind": kind, "content_type": "image/jpeg", "data": base64.b64encode(data).decode("ascii")}
                )
        return items

    def flush_evidence(self, force: bool = False) -> bool:
        """Send the pictures of crossings that lap_vision has acknowledged. Answers whether the work should go on.

        Pictures are a help to the person reviewing, not part of the result, so nothing here may hold up
        events, heartbeats or the final revision. When lap_vision cannot be reached for them the attempt is
        given up for a while and made again later; `force` makes it now. A picture that is missing on disk,
        or that lap_vision will not take, is let go: a crossing without a picture is still a crossing.
        """
        if not force and time.monotonic() < self._evidence_retry_at:
            return True
        while True:
            job = self._store.get_job(self._session_id)
            events = self._store.events_awaiting_evidence(self._session_id, job.acked_seq, PICTURES_PER_BATCH)
            if not events:
                self._evidence_failures = 0
                return True
            items = [item for event in events for item in self._pictures_of(event)]
            if items:
                try:
                    ack = self._transport.post(
                        f"{self._prefix}/evidence",
                        {"schema_version": SCHEMA_VERSION, "run_id": self._run_id, "items": items},
                    )
                except Rejected as refused:
                    log.warning("pictures were refused and are dropped: %s", refused)
                    ack = {}
                except DeliveryError as failure:
                    self._evidence_failures += 1
                    self._evidence_retry_at = time.monotonic() + min(2.0**self._evidence_failures, 60.0)
                    log.warning("pictures could not be sent, trying again later: %s", failure)
                    return True
                if ack.get("stop"):
                    return False
            self._store.mark_evidence_done(self._session_id, [event["seq"] for event in events])

    def heartbeat(self, state: str, media_position_ms: int, frames: int, fps: float, backlog_ms: int, error: str = "") -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self._run_id,
            "state": state,
            "media_position_ms": media_position_ms,
            "frames_processed": frames,
            "inference_fps": round(fps, 2),
            "backlog_ms": backlog_ms,
            "last_seq": self._store.last_seq(self._session_id),
            "outbox_pending": self._store.pending_count(self._session_id),
        }
        if error:
            payload["error"] = error[:500]
        return self._transport.post(f"{self._prefix}/heartbeat", payload)

    def revision(self, submission: dict[str, Any]) -> dict[str, Any]:
        return self._transport.post(f"{self._prefix}/revisions", submission)


def _wire(event: dict[str, Any]) -> dict[str, Any]:
    wire: dict[str, Any] = {
        "event_id": event["event_id"],
        "seq": event["seq"],
        "kind": event["kind"],
        "media_time_ms": event["media_time_ms"],
        "segment_id": event["segment_id"],
    }
    for key in ("participant_id", "plate_text", "identity_source", "evidence_ref"):
        if event[key]:
            wire[key] = event[key]
    if event["plate_confidence"] is not None:
        wire["plate_confidence"] = event["plate_confidence"]
    if event["bbox"] is not None:
        wire["bbox"] = event["bbox"]
    if event["extra"] is not None:
        wire["extra"] = event["extra"]
    return wire
