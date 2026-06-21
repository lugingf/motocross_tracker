"""Tests for pure Python logic in pipeline.py.

Covers: digit grouping, PlateRead construction, TrackState voting, event flattening.
No YOLO model is loaded — all tests use synthetic digit dicts or mock helpers.
"""
import pytest

from mx_tracker.pipeline import (
    PlateObservation,
    PlateRead,
    TrackState,
    _flatten_event,
    _group_score,
    _group_to_plate_read,
)


# ---------------------------------------------------------------------------
# _group_score — ranks groups by (length, avg_confidence)
# ---------------------------------------------------------------------------

class TestGroupScore:
    def _digit(self, conf: float) -> dict:
        return {"digit": "1", "conf": conf, "x_center": 0.0, "width": 10.0,
                "x1": 0.0, "x2": 10.0, "y1": 0.0, "y2": 20.0}

    def test_longer_group_beats_shorter_regardless_of_confidence(self):
        three_digits = [self._digit(0.5), self._digit(0.5), self._digit(0.5)]
        two_digits   = [self._digit(0.99), self._digit(0.99)]
        assert _group_score(three_digits) > _group_score(two_digits)

    def test_same_length_higher_avg_conf_wins(self):
        high_conf = [self._digit(0.9), self._digit(0.9)]
        low_conf  = [self._digit(0.5), self._digit(0.5)]
        assert _group_score(high_conf) > _group_score(low_conf)


# ---------------------------------------------------------------------------
# _group_to_plate_read — converts a digit group to PlateRead
# ---------------------------------------------------------------------------

class TestGroupToPlateRead:
    def _digit(self, d: str, conf: float, x: float) -> dict:
        return {"digit": d, "conf": conf, "x_center": x,
                "width": 10.0, "x1": x - 5, "x2": x + 5, "y1": 0.0, "y2": 20.0}

    def test_digits_joined_in_left_to_right_order(self):
        group = [self._digit("1", 0.9, 10), self._digit("3", 0.8, 30), self._digit("3", 0.7, 50)]
        result = _group_to_plate_read(group, plate_bbox=None, min_digits=1)
        assert result is not None
        assert result.text == "133"

    def test_x_center_is_average_of_digit_centers(self):
        group = [self._digit("4", 0.9, 20), self._digit("4", 0.9, 40)]
        result = _group_to_plate_read(group, plate_bbox=None, min_digits=1)
        assert result is not None
        assert result.x_center == pytest.approx(30.0)

    def test_confidence_is_average_of_digit_confidences(self):
        group = [self._digit("9", 0.8, 10), self._digit("9", 0.6, 30)]
        result = _group_to_plate_read(group, plate_bbox=None, min_digits=1)
        assert result is not None
        assert result.confidence == pytest.approx(0.7)

    def test_plate_bbox_passed_through(self):
        group = [self._digit("1", 0.9, 10)]
        bbox = (5, 5, 50, 30)
        result = _group_to_plate_read(group, plate_bbox=bbox, min_digits=1)
        assert result is not None
        assert result.plate_bbox == bbox

    def test_returns_none_when_number_shorter_than_min_digits(self):
        group = [self._digit("7", 0.9, 10)]  # 1 digit
        result = _group_to_plate_read(group, plate_bbox=None, min_digits=2)
        assert result is None

    def test_single_digit_accepted_when_min_digits_is_1(self):
        group = [self._digit("7", 0.9, 10)]
        result = _group_to_plate_read(group, plate_bbox=None, min_digits=1)
        assert result is not None
        assert result.text == "7"


# ---------------------------------------------------------------------------
# TrackState — plate observation voting
# ---------------------------------------------------------------------------

class TestTrackStateBestPlate:
    def _obs(self, text: str, conf: float, ts: float, frame: int = 0) -> PlateObservation:
        return PlateObservation(text=text, confidence=conf, timestamp=ts, frame_index=frame)

    def test_returns_none_when_no_observations(self):
        state = TrackState()
        assert state.best_plate(current_ts=10.0, vote_window_sec=5.0) is None

    def test_returns_the_only_observation_in_window(self):
        state = TrackState()
        state.add_observation(self._obs("83", 0.9, ts=5.0), ttl_sec=10.0)
        result = state.best_plate(current_ts=6.0, vote_window_sec=5.0)
        assert result is not None
        assert result.text == "83"

    def test_observation_outside_vote_window_ignored(self):
        state = TrackState()
        state.add_observation(self._obs("83", 0.9, ts=1.0), ttl_sec=100.0)
        # vote_window=2s, current_ts=10.0 → ts=1.0 is 9s ago → outside window
        result = state.best_plate(current_ts=10.0, vote_window_sec=2.0)
        # Falls back to stable_plate which was set
        assert result is not None and result.text == "83"

    def test_most_frequent_plate_wins_over_higher_single_confidence(self):
        state = TrackState()
        # "27" seen 3 times with moderate confidence
        for t in [1.0, 2.0, 3.0]:
            state.add_observation(self._obs("27", 0.7, ts=t), ttl_sec=10.0)
        # "9" seen once with very high confidence
        state.add_observation(self._obs("9", 0.99, ts=3.5), ttl_sec=10.0)
        result = state.best_plate(current_ts=4.0, vote_window_sec=10.0)
        assert result is not None
        assert result.text == "27"

    def test_prune_removes_observations_older_than_ttl(self):
        state = TrackState()
        state.add_observation(self._obs("27", 0.9, ts=1.0), ttl_sec=5.0)
        state.add_observation(self._obs("27", 0.9, ts=2.0), ttl_sec=5.0)
        # Prune at ts=10 with ttl=5 → all observations older than ts=5 removed
        state.prune(current_ts=10.0, ttl_sec=5.0)
        assert len(state.observations) == 0

    def test_recent_observations_survive_prune(self):
        state = TrackState()
        state.add_observation(self._obs("133", 0.8, ts=8.0), ttl_sec=5.0)
        state.add_observation(self._obs("133", 0.9, ts=9.0), ttl_sec=5.0)
        state.prune(current_ts=10.0, ttl_sec=5.0)
        # ts=8 is only 2s old → survives; ts=9 is 1s old → survives
        assert len(state.observations) == 2

    def test_duplicate_digits_resolved_by_higher_confidence(self):
        # Two reads of "133" at the same timestamp with different conf — higher should be kept
        state = TrackState()
        state.add_observation(self._obs("133", 0.6, ts=1.0), ttl_sec=10.0)
        state.add_observation(self._obs("133", 0.95, ts=2.0), ttl_sec=10.0)
        result = state.best_plate(current_ts=3.0, vote_window_sec=10.0)
        assert result is not None
        assert result.confidence == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# _flatten_event — converts nested bbox/center to flat CSV columns
# ---------------------------------------------------------------------------

class TestFlattenEvent:
    def _base_event(self) -> dict:
        return {
            "timestamp": 12.345,
            "rider_id": "plate_133",
            "identity_source": "plate",
            "bbox": [10, 20, 50, 80],
            "center": [30, 50],
            "crop_file": "/some/path.jpg",
        }

    def test_bbox_unpacked_to_individual_columns(self):
        result = _flatten_event(self._base_event())
        assert result["bbox_x1"] == 10
        assert result["bbox_y1"] == 20
        assert result["bbox_x2"] == 50
        assert result["bbox_y2"] == 80

    def test_center_unpacked_to_individual_columns(self):
        result = _flatten_event(self._base_event())
        assert result["center_x"] == 30
        assert result["center_y"] == 50

    def test_bbox_and_center_keys_removed(self):
        result = _flatten_event(self._base_event())
        assert "bbox" not in result
        assert "center" not in result

    def test_other_fields_pass_through_unchanged(self):
        result = _flatten_event(self._base_event())
        assert result["timestamp"] == pytest.approx(12.345)
        assert result["rider_id"] == "plate_133"
        assert result["crop_file"] == "/some/path.jpg"
