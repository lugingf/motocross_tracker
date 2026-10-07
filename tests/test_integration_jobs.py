import threading

import pytest

from mx_tracker.integration.delivery import DeliveryError, RunEnded
from mx_tracker.integration.jobs import JobError, source_url
from tests.integration_support import (
    OTHER_RUN,
    RUN,
    SESSION,
    FakeDetector,
    FakeTransport,
    job_request,
    make_manager,
    wait_for,
)

CROSSINGS = [(100.0, "plate_75"), (190.0, "plate_75"), (280.0, "plate_75")]


def start(tmp_path, detector=None, transport=None, **overrides):
    detector = detector or FakeDetector(CROSSINGS)
    transport = transport or FakeTransport()
    manager = make_manager(tmp_path, detector, transport, **overrides)
    return manager, detector, transport


def test_a_job_stores_delivers_and_acknowledges_what_the_detector_sees(tmp_path):
    manager, detector, transport = start(tmp_path)
    try:
        manager.create(job_request())

        assert wait_for(lambda: manager.store.get_job(SESSION).acked_seq == 3)
        batches = transport.paths("/events")
        events = [event for batch in batches for event in batch["events"]]
        assert [e["seq"] for e in events] == [1, 2, 3]
        assert all(batch["run_id"] == RUN for batch in batches)
        assert events[0]["participant_id"] == "plate_75" and events[0]["media_time_ms"] == 100000
        assert events[0]["bbox"] == {"x": 10, "y": 20, "w": 100, "h": 200}
        assert wait_for(lambda: transport.paths("/heartbeat"))
    finally:
        manager.close()


def test_the_detector_runs_with_the_persons_line_and_credentials_only_in_the_source(tmp_path):
    manager, detector, _ = start(tmp_path)
    try:
        manager.create(job_request())
        assert detector.started.wait(5)

        call = detector.calls[0]
        assert call["source"] == "rtsp://r-5d0f6c9e:read-secret@127.0.0.1:18554/imports/5d0f6c9e"
        assert call["settings"].line.value == "50.0000%,5.0000%,50.0000%,95.0000%"
        assert call["settings"].line.direction == "left_to_right"
        assert call["settings"].output.write_video is False
        assert call["settings"].output.save_plate_crops is True
    finally:
        manager.close()


def test_asking_for_the_same_run_again_answers_with_the_job_that_exists(tmp_path):
    manager, detector, _ = start(tmp_path)
    try:
        manager.create(job_request())
        assert detector.started.wait(5)

        again = manager.create(job_request())

        assert again["run_id"] == RUN and again["job_id"] == RUN
        assert len(detector.calls) == 1
    finally:
        manager.close()


def test_a_provisional_revision_follows_the_events_and_never_runs_ahead_of_them(tmp_path):
    manager, _, transport = start(tmp_path)
    try:
        manager.create(job_request())

        assert wait_for(lambda: transport.paths("/revisions"))
        revision = transport.paths("/revisions")[0]
        assert revision["status"] == "provisional"
        assert revision["manifest"]["last_seq"] <= manager.store.get_job(SESSION).acked_seq
        assert revision["manifest"]["event_count"] == revision["manifest"]["last_seq"]
        assert [lap["lap_time_ms"] for lap in revision["laps"]][:1] == [87500]
    finally:
        manager.close()


def test_finish_ends_the_detection_and_sends_a_final_revision_only_when_everything_is_acknowledged(tmp_path):
    manager, _, transport = start(tmp_path)
    try:
        manager.create(job_request())
        assert wait_for(lambda: manager.store.get_job(SESSION).acked_seq == 3)

        manager.finish(RUN)

        assert wait_for(lambda: manager.store.get_job(SESSION).state == "finished")
        final = [r for r in transport.paths("/revisions") if r["status"] == "final"]
        assert len(final) == 1
        assert final[0]["manifest"] == {"last_seq": 3, "event_count": 3, "gaps": []}
        assert len(final[0]["laps"]) == 3
    finally:
        manager.close()


def test_a_final_revision_waits_while_the_receiver_is_unreachable(tmp_path):
    manager, _, transport = start(tmp_path)
    try:
        manager.create(job_request())
        assert wait_for(lambda: manager.store.get_job(SESSION).acked_seq == 3)
        transport.fail = DeliveryError("down")

        manager.finish(RUN)
        assert wait_for(lambda: manager.store.get_job(SESSION).state == "finishing")
        assert not [r for r in transport.paths("/revisions") if r["status"] == "final"]

        transport.fail = None
        assert wait_for(lambda: manager.store.get_job(SESSION).state == "finished")
    finally:
        manager.close()


def test_events_are_kept_and_resent_after_the_receiver_was_unreachable(tmp_path):
    transport = FakeTransport()
    transport.fail = DeliveryError("down")
    manager, _, _ = start(tmp_path, transport=transport)
    try:
        manager.create(job_request())
        assert wait_for(lambda: manager.store.last_seq(SESSION) == 3)
        assert manager.store.get_job(SESSION).acked_seq == 0

        transport.fail = None

        assert wait_for(lambda: manager.store.get_job(SESSION).acked_seq == 3)
    finally:
        manager.close()


def test_a_gap_in_what_the_receiver_stored_is_filled_before_it_moves_on(tmp_path):
    transport = FakeTransport()
    original = transport.post
    dropped = {"done": False}

    def lossy(path, payload):
        if path.endswith("/events") and not dropped["done"] and len(payload["events"]) > 1:
            dropped["done"] = True
            payload = {**payload, "events": payload["events"][:1] + payload["events"][2:]}
        return original(path, payload)

    transport.post = lossy
    manager, _, _ = start(tmp_path, transport=transport)
    try:
        manager.create(job_request())

        assert wait_for(lambda: manager.store.get_job(SESSION).acked_seq == 3)
    finally:
        manager.close()


def test_when_the_receiver_says_stop_the_job_stops_and_does_not_finish(tmp_path):
    transport = FakeTransport()
    transport.stop_after = 1
    manager, _, _ = start(tmp_path, transport=transport)
    try:
        manager.create(job_request())

        assert wait_for(lambda: manager.store.get_job(SESSION).state == "stopped")
        assert not [r for r in transport.paths("/revisions") if r["status"] == "final"]
    finally:
        manager.close()


def test_a_token_the_receiver_no_longer_accepts_ends_the_job(tmp_path):
    transport = FakeTransport()
    transport.fail = RunEnded("gone")
    manager, _, _ = start(tmp_path, transport=transport)
    try:
        manager.create(job_request())

        assert wait_for(lambda: manager.store.get_job(SESSION).state == "stopped")
    finally:
        manager.close()


def test_a_detector_that_fails_is_reported_as_failed_and_keeps_its_events(tmp_path):
    transport = FakeTransport()
    manager, _, _ = start(tmp_path, detector=FakeDetector(CROSSINGS[:2], fail="Cannot open source: rtsp://127.0.0.1:18554/imports/5d0f6c9e"), transport=transport)
    try:
        manager.create(job_request())

        assert wait_for(lambda: manager.store.get_job(SESSION).state == "failed")
        assert wait_for(lambda: any(h["state"] == "failed" for h in transport.paths("/heartbeat")))
        assert "read-secret" not in manager.store.get_job(SESSION).error
    finally:
        manager.close()


def test_a_new_run_replaces_the_old_one_and_keeps_its_events(tmp_path):
    manager, detector, _ = start(tmp_path)
    try:
        manager.create(job_request())
        assert wait_for(lambda: manager.store.get_job(SESSION).acked_seq == 3)

        replaced = manager.create(job_request(run_id=OTHER_RUN, idempotency_key=OTHER_RUN))

        assert replaced["run_id"] == OTHER_RUN
        assert wait_for(lambda: len(detector.calls) == 2)
        assert manager.store.last_seq(SESSION) >= 3
    finally:
        manager.close()


def test_a_second_session_is_refused_while_the_tracker_is_busy(tmp_path):
    manager, detector, _ = start(tmp_path)
    try:
        manager.create(job_request())
        assert detector.started.wait(5)

        with pytest.raises(JobError) as refused:
            manager.create(job_request(session_id="6e0f6c9e-6a3c-4c35-9a3e-0b3f0f3a0009"))

        assert refused.value.status == 503
        assert manager.health()["state"] == "busy"
    finally:
        manager.close()


@pytest.mark.parametrize(
    "change",
    [
        {"source": {"url": "rtsp://evil.example/imports/x", "transport": "tcp"}},
        {"source": {"url": "/etc/passwd", "transport": "tcp"}},
        {"callback": {"base_url": "https://evil.example/api", "token": "t"}},
        {"callback": {"base_url": "https://lapvision.org/api", "token": ""}},
        {"session_id": "not-a-uuid"},
        {"schema_version": 99},
        {"configuration": {"version": 1}},
    ],
)
def test_a_request_outside_what_is_allowed_is_refused(tmp_path, change):
    manager, detector, _ = start(tmp_path)
    try:
        with pytest.raises(JobError):
            manager.create(job_request(**change))
        assert not detector.calls
    finally:
        manager.close()


def test_credentials_in_the_address_are_refused(tmp_path):
    with pytest.raises(JobError):
        source_url({"url": "rtsp://user:pw@127.0.0.1:18554/imports/x"})


def test_a_restart_marks_running_jobs_interrupted_and_a_new_start_resumes_on_the_same_timeline(tmp_path):
    manager, detector, transport = start(tmp_path)
    manager.create(job_request())
    assert wait_for(lambda: manager.store.get_job(SESSION).acked_seq == 3)
    manager.close()

    detector = FakeDetector([(5.0, "plate_75")])
    manager = make_manager(tmp_path, detector, transport)
    try:
        assert manager.store.get_job(SESSION).state == "interrupted"
        manager.create(job_request())

        assert wait_for(lambda: manager.store.last_seq(SESSION) == 5)
        events = manager.store.events_after(SESSION, 3, 10)
        assert events[0]["kind"] == "gap" and events[0]["extra"]["reason"] == "tracker_restart"
        assert events[0]["media_time_ms"] == 280000
        assert events[1]["media_time_ms"] >= events[0]["extra"]["to_ms"]
        assert events[1]["segment_id"] == events[0]["segment_id"] > 0
        assert wait_for(lambda: manager.store.get_job(SESSION).acked_seq == 5)
    finally:
        manager.close()


def test_recompute_on_a_running_job_produces_a_revision_with_the_corrections(tmp_path):
    manager, _, transport = start(tmp_path, revision_interval_sec=3600)
    try:
        manager.create(job_request())
        assert wait_for(lambda: manager.store.get_job(SESSION).acked_seq == 3)

        manager.recompute(
            RUN,
            {
                "callback": {"base_url": "https://lapvision.org/api", "token": "t"},
                "configuration": {"version": 4, "roster": []},
                "race_start_media_ms": None,
                "up_to_correction_id": 7,
                "corrections": [{"id": 7, "kind": "exclude", "event_ids": [f"{RUN[:8]}-3"]}],
            },
        )

        assert wait_for(lambda: any(r["applied_correction_id"] == 7 for r in transport.paths("/revisions")))
        revision = [r for r in transport.paths("/revisions") if r["applied_correction_id"] == 7][0]
        assert revision["config_version"] == 4
        assert revision["race_start_media_ms"] is None
        assert len(revision["laps"]) == 1
    finally:
        manager.close()


def test_recompute_on_a_finished_job_sends_a_final_revision_without_any_video(tmp_path):
    manager, _, transport = start(tmp_path)
    try:
        manager.create(job_request())
        assert wait_for(lambda: manager.store.get_job(SESSION).acked_seq == 3)
        manager.finish(RUN)
        assert wait_for(lambda: manager.store.get_job(SESSION).state == "finished")
        before = len(transport.paths("/revisions"))

        manager.recompute(
            RUN,
            {
                "callback": {"base_url": "https://lapvision.org/api", "token": "t"},
                "configuration": {"version": 5},
                "race_start_media_ms": 10000,
                "up_to_correction_id": 2,
                "corrections": [{"id": 2, "kind": "exclude", "event_ids": [f"{RUN[:8]}-1"]}],
            },
        )

        assert wait_for(lambda: len(transport.paths("/revisions")) > before)
        last = transport.paths("/revisions")[-1]
        assert last["status"] == "final" and last["applied_correction_id"] == 2
        assert len(last["laps"]) == 2
    finally:
        manager.close()


def test_stop_ends_the_job_and_is_idempotent(tmp_path):
    manager, _, _ = start(tmp_path)
    try:
        manager.create(job_request())
        assert wait_for(lambda: manager.store.get_job(SESSION).acked_seq == 3)

        assert manager.stop(RUN)["state"] == "stopped"
        assert manager.stop(RUN)["state"] == "stopped"
    finally:
        manager.close()


def test_an_unknown_job_is_not_found(tmp_path):
    manager, _, _ = start(tmp_path)
    try:
        with pytest.raises(JobError) as missing:
            manager.finish(RUN)
        assert missing.value.status == 404
    finally:
        manager.close()


CALLBACK = {"base_url": "https://lapvision.org/api", "token": "t"}


def test_a_lost_acknowledgement_leads_to_the_same_revision_being_sent_again(tmp_path):
    transport = FakeTransport()
    original = transport.post
    lost = {"done": False}

    def lossy(path, payload):
        answer = original(path, payload)
        if path.endswith("/revisions") and not lost["done"]:
            lost["done"] = True
            raise DeliveryError("the answer never arrived")
        return answer

    transport.post = lossy
    manager, _, _ = start(tmp_path, transport=transport)
    try:
        manager.create(job_request())
        assert wait_for(lambda: manager.store.get_job(SESSION).acked_seq == 3)
        manager.finish(RUN)

        assert wait_for(lambda: manager.store.get_job(SESSION).state == "finished")
        final = [r for r in transport.paths("/revisions") if r["status"] == "final"]
        assert len(final) >= 2
        assert len({r["submission_id"] for r in final}) == 1
        assert not manager.store.pending_submissions(SESSION)
    finally:
        manager.close()


def test_a_revision_that_was_not_acknowledged_is_still_there_after_a_restart(tmp_path):
    transport = FakeTransport()
    manager, _, _ = start(tmp_path, transport=transport)
    manager.create(job_request())
    assert wait_for(lambda: manager.store.get_job(SESSION).acked_seq == 3)
    transport.fail = DeliveryError("down")
    manager.finish(RUN)
    assert wait_for(lambda: manager.store.pending_submissions(SESSION))
    prepared = manager.store.pending_submissions(SESSION)[0]["submission_id"]
    manager.close()

    transport.fail = None
    manager = make_manager(tmp_path, FakeDetector(), transport)
    try:
        manager.finish(RUN, {"callback": CALLBACK})

        assert wait_for(lambda: manager.store.get_job(SESSION).state == "finished")
        assert prepared in {r["submission_id"] for r in transport.paths("/revisions")}
    finally:
        manager.close()


def test_a_failure_after_finish_is_not_reported_as_a_finished_job(tmp_path):
    transport = FakeTransport()
    detector = FakeDetector(CROSSINGS, fail_on_stop="the detector crashed")
    manager, _, _ = start(tmp_path, detector=detector, transport=transport)
    try:
        manager.create(job_request())
        assert wait_for(lambda: manager.store.get_job(SESSION).acked_seq == 3)

        manager.finish(RUN)

        assert wait_for(lambda: manager.store.get_job(SESSION).state == "failed")
        assert "the detector crashed" in manager.store.get_job(SESSION).error
        assert wait_for(lambda: any(h["state"] == "failed" and "crashed" in h.get("error", "") for h in transport.paths("/heartbeat")))
        assert not [r for r in transport.paths("/revisions") if r["status"] == "final"]
    finally:
        manager.close()


def test_a_full_outbox_after_finish_is_a_failure_too(tmp_path):
    transport = FakeTransport()
    transport.fail = DeliveryError("down")
    manager, _, _ = start(tmp_path, transport=transport, max_outbox_events=2)
    try:
        manager.create(job_request())
        assert wait_for(lambda: manager.store.last_seq(SESSION) >= 2)
        manager.finish(RUN)

        assert wait_for(lambda: manager.store.get_job(SESSION).state == "failed")
        assert "outbox" in manager.store.get_job(SESSION).error
    finally:
        manager.close()


def test_finishing_needs_no_stream_when_the_job_is_not_running(tmp_path):
    transport = FakeTransport()
    manager, _, _ = start(tmp_path, transport=transport)
    manager.create(job_request())
    assert wait_for(lambda: manager.store.get_job(SESSION).acked_seq == 3)
    manager.close()

    detector = FakeDetector()
    manager = make_manager(tmp_path, detector, transport)
    try:
        assert manager.store.get_job(SESSION).state == "interrupted"
        manager.finish(RUN, {"callback": CALLBACK})

        assert wait_for(lambda: manager.store.get_job(SESSION).state == "finished")
        assert not detector.calls
        assert [r for r in transport.paths("/revisions") if r["status"] == "final"]
    finally:
        manager.close()


def test_finishing_a_job_that_is_not_running_needs_an_allowed_callback(tmp_path):
    manager, _, transport = start(tmp_path)
    manager.create(job_request())
    assert wait_for(lambda: manager.store.get_job(SESSION).acked_seq == 3)
    manager.stop(RUN)
    manager.store.set_state(SESSION, "interrupted")
    try:
        with pytest.raises(JobError) as refused:
            manager.finish(RUN, {"callback": {"base_url": "https://evil.example", "token": "t"}})
        assert refused.value.status == 403
        with pytest.raises(JobError) as missing:
            manager.finish(RUN)
        assert missing.value.status == 403
    finally:
        manager.close()


def test_a_start_of_a_job_whose_finish_was_already_asked_does_not_read_the_stream(tmp_path):
    transport = FakeTransport()
    manager, _, _ = start(tmp_path, transport=transport)
    manager.create(job_request())
    assert wait_for(lambda: manager.store.get_job(SESSION).acked_seq == 3)
    manager.store.request_finish(SESSION)
    manager.close()

    detector = FakeDetector()
    manager = make_manager(tmp_path, detector, transport)
    try:
        manager.create(job_request())

        assert wait_for(lambda: manager.store.get_job(SESSION).state == "finished")
        assert not detector.calls
    finally:
        manager.close()
