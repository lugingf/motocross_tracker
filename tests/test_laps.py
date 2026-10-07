from mx_tracker.laps import Crossing, Gap, compute_laps


def cross(key, ts, rider="plate_7", number=7, confidence=None):
    return Crossing(key=key, timestamp=ts, participant_id=rider, number=number, confidence=confidence)


def test_known_start_makes_the_first_pass_a_lap():
    result = compute_laps([cross("a", 100.0), cross("b", 190.5)], race_start=10.0)

    assert [(lap.lap_number, lap.lap_time) for lap in result.laps] == [(1, 90.0), (2, 90.5)]
    assert result.baselines == []


def test_unknown_start_makes_the_first_pass_a_baseline():
    result = compute_laps([cross("a", 100.0), cross("b", 190.0), cross("c", 281.0)], race_start=None)

    assert result.baselines == ["a"]
    assert [(lap.lap_number, lap.lap_time) for lap in result.laps] == [(1, 90.0), (2, 91.0)]
    assert "a" not in result.assignments


def test_a_crossing_too_close_to_the_last_is_a_duplicate_and_does_not_move_the_clock():
    result = compute_laps(
        [cross("a", 100.0), cross("dup", 103.0), cross("b", 190.0)], race_start=10.0, min_lap=20.0
    )

    assert result.ignored == ["dup"]
    assert [(lap.lap_number, lap.lap_time) for lap in result.laps] == [(1, 90.0), (2, 90.0)]


def test_riders_are_counted_apart_and_ordered_by_laps_then_time():
    result = compute_laps(
        [
            cross("a1", 100.0, "plate_1", 1),
            cross("b1", 101.0, "plate_2", 2),
            cross("b2", 190.0, "plate_2", 2),
            cross("a2", 195.0, "plate_1", 1),
        ],
        race_start=10.0,
    )

    positions = {lap.keys[0]: lap.position for lap in result.laps}
    assert positions == {"a1": 1, "b1": 2, "b2": 1, "a2": 2}


def test_crossings_are_sorted_before_they_are_counted():
    result = compute_laps([cross("late", 190.0), cross("early", 100.0)], race_start=10.0)

    assert [lap.keys[0] for lap in result.laps] == ["early", "late"]


def test_a_lap_over_a_gap_is_flagged_and_not_mended():
    result = compute_laps(
        [cross("a", 100.0), cross("b", 190.0)], race_start=10.0, gaps=[Gap(start=120.0, end=150.0)]
    )

    assert [lap.quality for lap in result.laps] == [[], ["gap_before"]]
    assert result.laps[1].lap_time == 90.0


def test_a_lap_clear_of_every_gap_is_not_flagged():
    result = compute_laps([cross("a", 100.0), cross("b", 190.0)], race_start=10.0, gaps=[Gap(start=0.0, end=5.0)])

    assert all(lap.quality == [] for lap in result.laps)


def test_a_weak_read_is_flagged():
    result = compute_laps([cross("a", 100.0, confidence=0.3), cross("b", 190.0, confidence=0.9)], race_start=10.0, low_confidence=0.5)

    assert result.laps[0].quality == ["low_confidence"]
    assert result.laps[1].quality == []
