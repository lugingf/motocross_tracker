"""Tests for geometry.py — crossing detection, bbox expansion, zone crops, line parsing."""
import numpy as np
import pytest

from mx_tracker.geometry import (
    CrossingDecision,
    detect_crossing,
    expand_bbox,
    parse_line_arg,
    point_line_side_and_dist,
    zone_crop,
)


# ---------------------------------------------------------------------------
# parse_line_arg
# ---------------------------------------------------------------------------

class TestParseLineArg:
    def test_pixel_values_returned_as_is(self):
        x1, y1, x2, y2 = parse_line_arg("100,200,300,400", frame_width=1920, frame_height=1080)
        assert (x1, y1, x2, y2) == (100, 200, 300, 400)

    def test_percent_values_scaled_to_frame(self):
        x1, y1, x2, y2 = parse_line_arg("50%,5%,50%,95%", frame_width=1920, frame_height=1080)
        assert x1 == 960
        assert x2 == 960
        assert y1 == 54   # round(5/100 * 1080)
        assert y2 == 1026 # round(95/100 * 1080)

    def test_mixed_percent_and_pixel(self):
        x1, y1, x2, y2 = parse_line_arg("50%,0,50%,1080", frame_width=1920, frame_height=1080)
        assert x1 == 960
        assert y1 == 0
        assert y2 == 1080


# ---------------------------------------------------------------------------
# expand_bbox
# ---------------------------------------------------------------------------

class TestExpandBbox:
    def test_scale_1_returns_same_bbox(self):
        result = expand_bbox(100, 100, 200, 200, frame_width=1920, frame_height=1080, scale=1.0)
        assert result == (100, 100, 200, 200)

    def test_scale_2_doubles_size_around_center(self):
        x1, y1, x2, y2 = expand_bbox(100, 100, 200, 200, 1920, 1080, scale=2.0)
        # center=(150,150), original size 100x100, expanded to 200x200
        assert x1 == 50
        assert y1 == 50
        assert x2 == 250
        assert y2 == 250

    def test_clamped_to_frame_boundaries(self):
        # bbox touching the left edge — expanding left should stop at 0
        x1, y1, x2, y2 = expand_bbox(0, 0, 100, 100, frame_width=500, frame_height=500, scale=2.0)
        assert x1 >= 0
        assert y1 >= 0

    def test_clamped_to_right_and_bottom_boundaries(self):
        x1, y1, x2, y2 = expand_bbox(400, 400, 500, 500, frame_width=500, frame_height=500, scale=2.0)
        assert x2 <= 500
        assert y2 <= 500


# ---------------------------------------------------------------------------
# point_line_side_and_dist
# ---------------------------------------------------------------------------

class TestPointLineSideAndDist:
    def test_point_on_left_side_of_vertical_line(self):
        # vertical line at x=100
        side, dist = point_line_side_and_dist((50, 50), (100, 0), (100, 100))
        assert side > 0  # one side
        assert dist == pytest.approx(50.0)

    def test_point_on_right_side_of_vertical_line(self):
        side, dist = point_line_side_and_dist((150, 50), (100, 0), (100, 100))
        assert side < 0  # opposite side
        assert dist == pytest.approx(50.0)

    def test_point_on_line_has_zero_distance(self):
        _, dist = point_line_side_and_dist((100, 50), (100, 0), (100, 100))
        assert dist == pytest.approx(0.0, abs=1e-6)

    def test_sides_are_opposite_for_points_on_opposite_sides(self):
        side_left, _ = point_line_side_and_dist((50, 50), (100, 0), (100, 100))
        side_right, _ = point_line_side_and_dist((150, 50), (100, 0), (100, 100))
        assert side_left * side_right < 0


# ---------------------------------------------------------------------------
# detect_crossing
# ---------------------------------------------------------------------------

# Vertical finish line at x=500, y from 0 to 1000.
LINE_A = (500, 0)
LINE_B = (500, 1000)
LINE_WIDTH = 20


class TestDetectCrossingNoCrossing:
    def test_no_previous_position_never_counts_as_crossing(self):
        decision = detect_crossing(None, (400, 500), LINE_A, LINE_B, LINE_WIDTH, "left_to_right")
        assert decision.crossed is False

    def test_staying_on_same_side_is_not_a_crossing(self):
        # Both points left of the line
        decision = detect_crossing((300, 500), (400, 500), LINE_A, LINE_B, LINE_WIDTH, "left_to_right")
        assert decision.crossed is False

    def test_direction_filter_blocks_wrong_direction(self):
        # Bike crosses left→right: must be accepted by left_to_right, blocked by right_to_left
        prev = (400, 500)   # left of line
        cur  = (600, 500)   # right of line
        decision_ltr = detect_crossing(prev, cur, LINE_A, LINE_B, LINE_WIDTH, "left_to_right")
        decision_rtl = detect_crossing(prev, cur, LINE_A, LINE_B, LINE_WIDTH, "right_to_left")
        assert decision_ltr.crossed is True
        assert decision_rtl.crossed is False

    def test_right_to_left_crossing_accepted_by_rtl_only(self):
        # Bike crosses right→left: must be accepted by right_to_left, blocked by left_to_right
        prev = (600, 500)   # right of line
        cur  = (400, 500)   # left of line
        decision_ltr = detect_crossing(prev, cur, LINE_A, LINE_B, LINE_WIDTH, "left_to_right")
        decision_rtl = detect_crossing(prev, cur, LINE_A, LINE_B, LINE_WIDTH, "right_to_left")
        assert decision_ltr.crossed is False
        assert decision_rtl.crossed is True


class TestDetectCrossingWithCrossing:
    def test_crossing_detected_when_points_on_opposite_sides(self):
        prev = (400, 500)   # left
        cur  = (600, 500)   # right
        decision = detect_crossing(prev, cur, LINE_A, LINE_B, LINE_WIDTH, "left_to_right")
        # A crossing must have occurred (in one of the two modes)
        assert decision.crossed or detect_crossing(
            prev, cur, LINE_A, LINE_B, LINE_WIDTH, "right_to_left"
        ).crossed

    def test_crossing_sets_a_direction(self):
        prev = (400, 500)
        cur  = (600, 500)
        for mode in ("left_to_right", "right_to_left"):
            d = detect_crossing(prev, cur, LINE_A, LINE_B, LINE_WIDTH, mode)
            if d.crossed:
                assert d.direction in ("left_to_right", "right_to_left")

    def test_distance_is_non_negative(self):
        prev = (400, 500)
        cur  = (600, 500)
        d = detect_crossing(prev, cur, LINE_A, LINE_B, LINE_WIDTH, "left_to_right")
        assert d.distance >= 0


# ---------------------------------------------------------------------------
# zone_crop
# ---------------------------------------------------------------------------

class TestZoneCrop:
    def setup_method(self):
        # 600×300 image — 3 columns × 2 rows when using 6 zones, or 3×3 for 9 zones
        self.img = np.zeros((300, 600, 3), dtype=np.uint8)

    def test_n_zones_1_returns_full_image(self):
        result = zone_crop(self.img, n_zones=1, selected=[1])
        assert result.shape == self.img.shape

    def test_empty_selected_returns_full_image(self):
        result = zone_crop(self.img, n_zones=9, selected=[])
        assert result.shape == self.img.shape

    def test_unsupported_n_zones_returns_full_image(self):
        result = zone_crop(self.img, n_zones=7, selected=[1])
        assert result.shape == self.img.shape

    def test_bottom_right_zones_of_9_are_smaller_slice(self):
        # Zones 8,9 are the bottom-right 2 cells of a 3×3 grid
        result = zone_crop(self.img, n_zones=9, selected=[8, 9])
        assert result.shape[0] < self.img.shape[0]  # shorter height
        assert result.shape[1] < self.img.shape[1]  # narrower width

    def test_zone_crop_covers_correct_pixel_region(self):
        # Paint zone 5 (center of 3×3) red, verify the crop contains red
        h, w = self.img.shape[:2]
        zone_h, zone_w = h // 3, w // 3
        # Zone 5 is row=1, col=1
        self.img[zone_h:2*zone_h, zone_w:2*zone_w] = (0, 0, 255)
        result = zone_crop(self.img, n_zones=9, selected=[5])
        assert result[:, :, 2].max() == 255  # red channel present
