import json
import threading
import urllib.error
import urllib.request

import pytest

from mx_tracker.integration import api as api_module
from mx_tracker.integration.api import make_server
from tests.integration_support import (
    RUN,
    SESSION,
    FakeDetector,
    FakeTransport,
    job_request,
    make_manager,
    settings_for,
    wait_for,
)


@pytest.fixture
def served(tmp_path):
    transport = FakeTransport()
    detector = FakeDetector([(100.0, "plate_75")])
    manager = make_manager(tmp_path, detector, transport, port=0)
    server = make_server(manager.settings, manager)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    yield base, manager
    server.shutdown()
    server.server_close()
    manager.close()


def call(base, method, path, body=None, token="tracker-token"):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(base + path, data=data, method=method)
    if token is not None:
        request.add_header("Authorization", f"Bearer {token}")
    if data is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def test_every_route_needs_the_token(served):
    base, _ = served

    for token in (None, "wrong", ""):
        assert call(base, "GET", "/integration/v1/health", token=token)[0] == 401
        assert call(base, "POST", "/integration/v1/jobs", job_request(), token=token)[0] == 401


def test_health_says_whether_the_tracker_is_busy(served):
    base, _ = served

    status, body = call(base, "GET", "/integration/v1/health")

    assert status == 200
    assert json.loads(body)["state"] == "ready"


def test_a_job_can_be_created_read_finished_and_stopped(served):
    base, manager = served

    status, body = call(base, "POST", "/integration/v1/jobs", job_request())
    assert status == 202
    assert json.loads(body)["job_id"] == RUN

    assert call(base, "GET", f"/integration/v1/jobs/{RUN}")[0] == 200
    assert call(base, "POST", f"/integration/v1/jobs/{RUN}/finish", {})[0] == 202
    assert wait_for(lambda: manager.store.get_job(SESSION).state in ("finishing", "finished"))
    assert call(base, "POST", f"/integration/v1/jobs/{RUN}/stop", {})[0] == 200


def test_a_request_that_is_not_allowed_is_answered_with_a_reason(served):
    base, _ = served

    status, body = call(base, "POST", "/integration/v1/jobs", job_request(source={"url": "file:///etc/passwd", "transport": "tcp"}))

    assert status == 403
    assert json.loads(body)["error"]["message"] == "this source is not allowed"


def test_the_legacy_job_routes_are_not_on_this_listener(served):
    base, _ = served

    for method, path in (("GET", "/jobs"), ("POST", "/jobs"), ("GET", "/health"), ("POST", "/jobs/abc/stop")):
        assert call(base, "GET" if method == "GET" else "POST", path, {} if method == "POST" else None)[0] == 404


def test_an_unknown_job_is_404_and_a_bad_body_is_400(served):
    base, _ = served

    assert call(base, "GET", f"/integration/v1/jobs/{RUN}")[0] == 404
    request = urllib.request.Request(base + "/integration/v1/jobs", data=b"not json", method="POST")
    request.add_header("Authorization", "Bearer tracker-token")
    with pytest.raises(urllib.error.HTTPError) as bad:
        urllib.request.urlopen(request, timeout=5)
    assert bad.value.code == 400


def test_preview_returns_a_frame_for_an_allowed_source_only(served, monkeypatch):
    base, _ = served
    monkeypatch.setattr(api_module, "grab_frame", lambda source, settings: b"\xff\xd8jpeg")

    status, body = call(base, "POST", "/integration/v1/preview", {"source": {"url": "rtsp://127.0.0.1:18554/imports/x"}})
    assert (status, body) == (200, b"\xff\xd8jpeg")
    assert call(base, "POST", "/integration/v1/preview", {"source": {"url": "rtsp://evil.example/x"}})[0] == 403


def test_a_preview_failure_does_not_leak_the_credentials(served, monkeypatch):
    base, _ = served

    def broken(source, settings):
        raise RuntimeError("could not open rtsp://u:topsecret@127.0.0.1:18554/imports/x")

    monkeypatch.setattr(api_module, "grab_frame", broken)

    status, body = call(base, "POST", "/integration/v1/preview", {"source": {"url": "rtsp://127.0.0.1:18554/imports/x", "username": "u", "password": "topsecret"}})

    assert status == 502
    assert b"topsecret" not in body


def test_the_listener_refuses_to_start_without_a_token(tmp_path, monkeypatch):
    monkeypatch.delenv("MX_INTEGRATION_TOKEN", raising=False)
    manager = make_manager(tmp_path, FakeDetector(), FakeTransport(), token="")
    try:
        with pytest.raises(RuntimeError):
            make_server(manager.settings, manager)
    finally:
        manager.close()


def test_the_token_can_come_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("MX_INTEGRATION_TOKEN", "from-env")
    manager = make_manager(tmp_path, FakeDetector(), FakeTransport(), token="", port=0)
    try:
        server = make_server(manager.settings, manager)
        server.server_close()
    finally:
        manager.close()
