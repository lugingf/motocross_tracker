from mx_tracker.integration.engine import apply_corrections, build_revision, number_of
from tests.integration_support import fixture

CONFIG = {
    "version": 3,
    "calibration": {"min_lap_time_ms": 20000},
    "roster": [{"number": 75, "name": "Jane Doe", "team": "Team"}],
}


def crossing(seq, ts_ms, participant, source="plate", confidence=0.9):
    return {
        "event_id": f"e{seq}",
        "seq": seq,
        "kind": "crossing",
        "media_time_ms": ts_ms,
        "participant_id": participant,
        "plate_confidence": confidence,
        "identity_source": source,
        "extra": None,
    }


def gap(seq, from_ms, to_ms):
    return {"event_id": f"e{seq}", "seq": seq, "kind": "gap", "media_time_ms": from_ms, "participant_id": "", "extra": {"to_ms": to_ms}}


def revision(events, corrections=(), start=10000, **extra):
    return build_revision(events, list(corrections), CONFIG, start, 0.5, "run", "sub-1", "final", 0)


def test_number_of_reads_only_plate_participants():
    assert number_of("plate_75") == 75
    assert number_of("reid_3") is None
    assert number_of("") is None


def test_a_revision_has_the_laps_the_names_and_the_manifest():
    result = revision([crossing(1, 100000, "plate_75"), crossing(2, 190000, "plate_75")])

    assert [(lap["lap_number"], lap["lap_time_ms"], lap["driver_name"], lap["team"]) for lap in result["laps"]] == [
        (1, 90000, "Jane Doe", "Team"),
        (2, 90000, "Jane Doe", "Team"),
    ]
    assert result["manifest"] == {"last_seq": 2, "event_count": 2, "gaps": []}
    assert result["config_version"] == 3
    assert result["status"] == "final"


def test_unknown_start_leaves_the_first_pass_as_a_baseline():
    result = revision([crossing(1, 100000, "plate_75"), crossing(2, 190000, "plate_75")], start=None)

    assert [lap["lap_number"] for lap in result["laps"]] == [1]
    assert result["race_start_media_ms"] is None


def test_a_duplicate_inside_the_shortest_lap_is_dropped():
    result = revision([crossing(1, 100000, "plate_75"), crossing(2, 103000, "plate_75"), crossing(3, 190000, "plate_75")])

    assert [lap["lap_time_ms"] for lap in result["laps"]] == [90000, 90000]


def test_a_gap_flags_the_lap_it_lies_in_and_is_in_the_manifest():
    result = revision([crossing(1, 100000, "plate_75"), gap(2, 120000, 150000), crossing(3, 190000, "plate_75")])

    assert result["laps"][1]["quality"] == ["gap_before"]
    assert result["manifest"]["gaps"] == [{"from_ms": 120000, "to_ms": 150000, "reason": "reconnect"}]
    assert result["manifest"]["event_count"] == 3


def test_a_rider_without_a_number_is_flagged_and_a_crossing_without_a_rider_is_not_a_lap():
    result = revision([crossing(1, 100000, "reid_9", "reid"), crossing(2, 105000, "", "unknown"), crossing(3, 190000, "reid_9", "reid")])

    assert [lap["quality"] for lap in result["laps"]] == [["unknown_number"], ["unknown_number"]]
    assert all(lap["number"] is None for lap in result["laps"])


def test_a_weak_read_is_flagged():
    result = revision([crossing(1, 100000, "plate_75", confidence=0.2)])

    assert result["laps"][0]["quality"] == ["low_confidence"]


def test_set_number_gives_an_unread_crossing_to_a_rider():
    events = [crossing(1, 100000, "plate_75"), crossing(2, 190000, "", "unknown")]

    result = revision(events, [{"id": 1, "kind": "set_number", "event_ids": ["e2"], "number": 75}])

    assert [lap["lap_number"] for lap in result["laps"]] == [1, 2]
    assert result["laps"][1]["number"] == 75


def test_merge_moves_a_misread_number_onto_the_right_rider():
    events = [crossing(1, 100000, "plate_75"), crossing(2, 190000, "plate_57")]

    result = revision(events, [{"id": 1, "kind": "merge", "from_number": 57, "number": 75}])

    assert [(lap["participant_id"], lap["lap_number"]) for lap in result["laps"]] == [("plate_75", 1), ("plate_75", 2)]


def test_exclude_removes_a_crossing_and_include_brings_it_back():
    events = [crossing(1, 100000, "plate_75"), crossing(2, 190000, "plate_75")]
    exclude = {"id": 1, "kind": "exclude", "event_ids": ["e2"]}

    assert len(revision(events, [exclude])["laps"]) == 1
    assert len(revision(events, [exclude, {"id": 2, "kind": "include", "event_ids": ["e2"]}])["laps"]) == 2


def test_corrections_apply_in_the_order_they_were_made():
    events = [crossing(1, 100000, "plate_75")]
    corrections = [
        {"id": 2, "kind": "exclude", "event_ids": ["e1"]},
        {"id": 1, "kind": "set_number", "event_ids": ["e1"], "number": 9},
    ]

    adjusted = apply_corrections(events, corrections)

    assert adjusted == []


def test_the_originals_are_not_changed_by_a_correction():
    events = [crossing(1, 100000, "")]

    apply_corrections(events, [{"id": 1, "kind": "set_number", "event_ids": ["e1"], "number": 9}])

    assert events[0]["participant_id"] == ""


def test_a_revision_has_exactly_the_fields_of_the_contract():
    golden = fixture("revision_submission.json")
    result = revision([crossing(1, 100000, "plate_75")])

    assert set(result) - {"schema_version"} == set(golden) - {"schema_version"}
    assert set(result["manifest"]) == set(golden["manifest"])
    contract_lap_fields = {
        "participant_id", "number", "driver_name", "team", "lap_number", "lap_time_ms",
        "finish_media_ms", "position", "confidence", "quality", "event_ids",
    }
    assert set(golden["laps"][0]) <= contract_lap_fields
    assert set(result["laps"][0]) == contract_lap_fields
