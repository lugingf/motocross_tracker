from mx_tracker.integration.store import Store
from tests.integration_support import RUN, SESSION


def make(tmp_path):
    store = Store(tmp_path / "s.sqlite")
    store.create_job(SESSION, RUN, {"version": 1}, None)
    return store


def test_events_get_consecutive_sequence_numbers_and_ids_that_name_their_run(tmp_path):
    store = make(tmp_path)

    first = store.append_event(SESSION, RUN, "crossing", 1000, 0, participant_id="plate_7")
    second = store.append_event(SESSION, RUN, "crossing", 2000, 0, participant_id="plate_7")

    assert (first, second) == (1, 2)
    assert [e["event_id"] for e in store.events_after(SESSION, 0, 10)] == [f"{RUN[:8]}-1", f"{RUN[:8]}-2"]
    assert store.last_seq(SESSION) == 2


def test_pending_is_what_the_receiver_has_not_acknowledged(tmp_path):
    store = make(tmp_path)
    for index in range(3):
        store.append_event(SESSION, RUN, "crossing", index * 1000, 0)

    assert store.pending_count(SESSION) == 3
    store.acknowledge(SESSION, 2)
    assert store.pending_count(SESSION) == 1
    store.acknowledge(SESSION, 1)
    assert store.get_job(SESSION).acked_seq == 2


def test_events_survive_a_reopen_and_numbering_continues(tmp_path):
    store = make(tmp_path)
    store.append_event(SESSION, RUN, "crossing", 1000, 0)
    store.close()

    reopened = Store(tmp_path / "s.sqlite")

    assert reopened.last_seq(SESSION) == 1
    assert reopened.append_event(SESSION, RUN, "crossing", 2000, 0) == 2


def test_the_timeline_position_only_moves_forward(tmp_path):
    store = make(tmp_path)

    store.touch(SESSION, 5000)
    store.touch(SESSION, 3000)

    assert store.get_job(SESSION).last_media_ms == 5000
