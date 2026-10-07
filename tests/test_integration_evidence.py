import base64

import cv2
import numpy as np

from mx_tracker.integration.delivery import MAX_PICTURE_BYTES, fit_picture
from tests.integration_support import JPEG, RUN, SESSION, FakeDetector, FakeTransport, job_request, make_manager, wait_for


def sent(transport):
    return [item for batch in transport.paths("/evidence") for item in batch["items"]]


def run_with_pictures(tmp_path, crossings, transport=None):
    transport = transport or FakeTransport()
    manager = make_manager(tmp_path, FakeDetector(crossings, pictures=True), transport)
    manager.create(job_request())
    return manager, transport


def test_the_plate_of_a_read_crossing_is_sent_once_the_crossing_is_acknowledged(tmp_path):
    manager, transport = run_with_pictures(tmp_path, [(100.0, "plate_75"), (190.0, "plate_75")])
    try:
        assert wait_for(lambda: len(sent(transport)) == 2)

        items = sent(transport)
        assert {item["kind"] for item in items} == {"plate"}
        assert {item["event_id"] for item in items} == {f"{RUN[:8]}-1", f"{RUN[:8]}-2"}
        assert base64.b64decode(items[0]["data"]) == JPEG
        assert items[0]["content_type"] == "image/jpeg"
    finally:
        manager.close()


def test_a_crossing_nobody_could_read_sends_the_rider_as_well(tmp_path):
    manager, transport = run_with_pictures(tmp_path, [(100.0, "unknown")])
    try:
        assert wait_for(lambda: len(sent(transport)) == 2)

        assert sorted(item["kind"] for item in sent(transport)) == ["bike", "plate"]
    finally:
        manager.close()


def test_a_picture_is_sent_only_once(tmp_path):
    manager, transport = run_with_pictures(tmp_path, [(100.0, "plate_75")])
    try:
        assert wait_for(lambda: len(sent(transport)) == 1)
        manager.finish(RUN)
        assert wait_for(lambda: manager.store.get_job(SESSION).state == "finished")

        assert len(sent(transport)) == 1
        assert not manager.store.events_awaiting_evidence(SESSION, 100, 10)
    finally:
        manager.close()


def test_pictures_are_not_sent_before_their_crossing_has_been_acknowledged(tmp_path):
    transport = FakeTransport()
    transport.stop_after = None
    original = transport.post

    def refuse_events(path, payload):
        if path.endswith("/events"):
            from mx_tracker.integration.delivery import DeliveryError

            raise DeliveryError("down")
        return original(path, payload)

    transport.post = refuse_events
    manager, _ = run_with_pictures(tmp_path, [(100.0, "plate_75")], transport)
    try:
        assert wait_for(lambda: manager.store.last_seq(SESSION) == 1)
        assert not sent(transport)
    finally:
        manager.close()


def test_a_missing_picture_does_not_hold_up_the_end_of_the_job(tmp_path):
    manager, transport = run_with_pictures(tmp_path, [(100.0, "plate_75")])
    try:
        assert wait_for(lambda: manager.store.get_job(SESSION).acked_seq == 1)
        for path in (tmp_path / "state").rglob("*.jpg"):
            path.unlink()
        manager.store._db.execute("UPDATE events SET evidence_done = 0")

        manager.finish(RUN)

        assert wait_for(lambda: manager.store.get_job(SESSION).state == "finished")
    finally:
        manager.close()


def test_pictures_lap_vision_refuses_are_dropped_and_the_job_still_finishes(tmp_path):
    transport = FakeTransport()
    transport.reject_evidence = True
    manager, _ = run_with_pictures(tmp_path, [(100.0, "plate_75")], transport)
    try:
        assert wait_for(lambda: manager.store.get_job(SESSION).acked_seq == 1)
        manager.finish(RUN)

        assert wait_for(lambda: manager.store.get_job(SESSION).state == "finished")
        assert not manager.store.events_awaiting_evidence(SESSION, 100, 10)
    finally:
        manager.close()


def test_a_large_picture_is_scaled_down_to_fit(tmp_path):
    noise = np.random.default_rng(1).integers(0, 255, (1200, 1600, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", noise, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok and len(encoded) > MAX_PICTURE_BYTES

    fitted = fit_picture(encoded.tobytes())

    assert fitted is not None and len(fitted) <= MAX_PICTURE_BYTES and fitted[:3] == b"\xff\xd8\xff"


def test_something_that_is_not_an_image_is_not_sent():
    assert fit_picture(b"x" * (MAX_PICTURE_BYTES + 10)) is None
    assert fit_picture(JPEG) == JPEG


def test_pictures_that_cannot_be_delivered_do_not_hold_up_events_or_the_final_revision(tmp_path):
    from mx_tracker.integration.delivery import DeliveryError

    transport = FakeTransport()
    original = transport.post

    def evidence_down(path, payload):
        if path.endswith("/evidence"):
            raise DeliveryError("lap_vision answered 503")
        return original(path, payload)

    transport.post = evidence_down
    manager, _ = run_with_pictures(tmp_path, [(100.0, "plate_75"), (190.0, "plate_75")], transport)
    try:
        assert wait_for(lambda: manager.store.get_job(SESSION).acked_seq == 2)
        manager.finish(RUN)

        assert wait_for(lambda: manager.store.get_job(SESSION).state == "finished")
        assert [r for r in transport.paths("/revisions") if r["status"] == "final"]
        assert wait_for(lambda: transport.paths("/heartbeat"))
    finally:
        manager.close()


def test_pictures_are_tried_again_after_lap_vision_comes_back(tmp_path):
    from mx_tracker.integration.delivery import DeliveryError

    transport = FakeTransport()
    original = transport.post
    down = {"yes": True}

    def flaky(path, payload):
        if path.endswith("/evidence") and down["yes"]:
            raise DeliveryError("down")
        return original(path, payload)

    transport.post = flaky
    manager, _ = run_with_pictures(tmp_path, [(100.0, "plate_75")], transport)
    try:
        assert wait_for(lambda: manager.store.get_job(SESSION).acked_seq == 1)
        down["yes"] = False
        manager.finish(RUN)

        assert wait_for(lambda: manager.store.get_job(SESSION).state == "finished")
        assert len(sent(transport)) == 1
    finally:
        manager.close()


def test_the_rider_is_sent_for_a_crossing_matched_without_a_number(tmp_path):
    from mx_tracker.integration.delivery import Delivery
    from mx_tracker.integration.store import Store

    store = Store(tmp_path / "s.sqlite")
    base = tmp_path / "run"
    (base / "unresolved").mkdir(parents=True)
    (base / "unresolved" / "f.jpg").write_bytes(JPEG)
    (base / "unresolved" / "f_bike.jpg").write_bytes(JPEG)
    delivery = Delivery(store, FakeTransport(), SESSION, RUN, 10, lambda run_id: base)
    event = {"event_id": "e1", "run_id": RUN, "evidence_ref": "unresolved/f.jpg", "participant_id": "reid_9", "identity_source": "reid", "plate_text": ""}

    kinds = sorted(item["kind"] for item in delivery._pictures_of(event))

    assert kinds == ["bike", "plate"]
    event.update(participant_id="plate_75", identity_source="plate", plate_text="75")
    assert [item["kind"] for item in delivery._pictures_of(event)] == ["plate"]
