"""Tests for motion.py and the entry-zone geometry it depends on.

The gate trades work for risk: every frame it skips is a frame no bike can be
found in. So the tests below care less about how much it skips than about what
it must never skip — a rider that has already moved past the entry zone, the
frames while the background is still settling, and anything at all when the
entry zone cannot be placed.
"""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from mx_tracker.config import TrackerSettings
from mx_tracker.geometry import entry_zone_rect
from mx_tracker.motion import GateDecision, MotionGate


FRAME_W, FRAME_H = 800, 400
# Vertical finish line down the middle, top to bottom.
LINE_A, LINE_B = (400, 0), (400, 400)


def _settings(**gate_overrides) -> TrackerSettings:
    settings = TrackerSettings()
    settings.line.direction = "left_to_right"
    settings.motion_gate.warmup_frames = 2
    settings.motion_gate.hold_sec = 0.0
    for key, value in gate_overrides.items():
        setattr(settings.motion_gate, key, value)
    return settings


def _gate(settings=None, line_a=LINE_A, line_b=LINE_B) -> MotionGate:
    return MotionGate(settings or _settings(), line_a, line_b, FRAME_W, FRAME_H)


def _frame(value=0):
    return np.full((FRAME_H, FRAME_W, 3), value, np.uint8)


def _frame_with_blob(x, y=200, size=120, value=255):
    """A frame with one bright square — stands in for a bike."""
    frame = _frame(0)
    frame[max(0, y - size // 2):y + size // 2, max(0, x - size // 2):x + size // 2] = value
    return frame


def _settle(gate, frames=6, value=0, start_ts=0.0, step=0.1):
    """Feed static frames until the gate is past warmup and reports idle."""
    ts = start_ts
    for _ in range(frames):
        decision = gate.should_process(_frame(value), ts)
        ts += step
    return decision, ts


# ---------------------------------------------------------------------------
# entry_zone_rect — which quarter of the frame riders come in through
# ---------------------------------------------------------------------------

class TestEntryZoneRect:
    def test_left_to_right_takes_the_left_strip(self):
        assert entry_zone_rect(LINE_A, LINE_B, "left_to_right", 800, 400, 0.25) == (0, 0, 200, 400)

    def test_right_to_left_takes_the_right_strip(self):
        assert entry_zone_rect(LINE_A, LINE_B, "right_to_left", 800, 400, 0.25) == (600, 0, 800, 400)

    def test_swapping_the_line_endpoints_swaps_the_entry_side(self):
        # `direction` is measured against the line's own a→b orientation, so a
        # line calibrated bottom-to-top expects riders from the other side.
        assert entry_zone_rect((400, 400), (400, 0), "left_to_right", 800, 400, 0.25) == (600, 0, 800, 400)

    def test_zone_fraction_sets_the_strip_width(self):
        assert entry_zone_rect(LINE_A, LINE_B, "left_to_right", 800, 400, 0.5) == (0, 0, 400, 400)

    def test_full_frame_fraction_covers_everything(self):
        assert entry_zone_rect(LINE_A, LINE_B, "left_to_right", 800, 400, 1.0) == (0, 0, 800, 400)

    def test_tiny_fraction_still_yields_at_least_one_pixel(self):
        rect = entry_zone_rect(LINE_A, LINE_B, "left_to_right", 800, 400, 0.0001)
        assert rect is not None and rect[2] - rect[0] >= 1

    def test_horizontal_line_uses_a_horizontal_strip(self):
        # Riders travel along y, so the strip spans the full width.
        rect = entry_zone_rect((0, 200), (800, 200), "left_to_right", 800, 400, 0.25)
        assert rect == (0, 300, 800, 400)

    def test_horizontal_line_other_direction_takes_the_opposite_end(self):
        rect = entry_zone_rect((0, 200), (800, 200), "right_to_left", 800, 400, 0.25)
        assert rect == (0, 0, 800, 100)

    def test_mostly_vertical_diagonal_still_uses_an_x_strip(self):
        rect = entry_zone_rect((350, 0), (450, 400), "left_to_right", 800, 400, 0.25)
        assert rect == (0, 0, 200, 400)

    def test_line_outside_the_frame_has_no_unambiguous_entry_side(self):
        assert entry_zone_rect((-100, 0), (-100, 400), "left_to_right", 800, 400, 0.25) is None

    def test_line_beyond_the_far_edge_keeps_the_incoming_strip(self):
        # The line is off-frame, so the whole frame sits on the incoming side
        # and nothing can cross at all — the entry strip is still the left
        # quarter, and gating there costs no crossings.
        assert entry_zone_rect((900, 0), (900, 400), "left_to_right", 800, 400, 0.25) == (0, 0, 200, 400)

    def test_line_behind_the_near_edge_has_no_entry_side(self):
        # Here the whole frame is on the *outgoing* side, so there is nowhere
        # for riders to come in from and gating would be guesswork.
        assert entry_zone_rect((-100, 0), (-100, 400), "left_to_right", 800, 400, 0.25) is None


# ---------------------------------------------------------------------------
# Wiring and self-disabling
# ---------------------------------------------------------------------------

class TestGateSetup:
    def test_gate_is_active_for_a_normal_line(self):
        assert _gate().active is True

    def test_zone_matches_the_configured_direction(self):
        assert _gate().zone == (0, 0, 200, 400)

    def test_disabled_in_settings_stays_inactive(self):
        gate = _gate(_settings(enabled=False))
        assert gate.active is False
        assert "disabled in settings" in gate.describe()

    def test_disabled_gate_processes_every_frame(self):
        gate = _gate(_settings(enabled=False))
        for index in range(20):
            assert gate.should_process(_frame(0), index * 0.1).process is True

    def test_ambiguous_entry_zone_disables_the_gate(self):
        gate = _gate(line_a=(-100, 0), line_b=(-100, 400))
        assert gate.active is False
        assert "ambiguous" in gate.describe()

    def test_ambiguous_zone_gate_never_skips_a_frame(self):
        gate = _gate(line_a=(-100, 0), line_b=(-100, 400))
        for index in range(20):
            assert gate.should_process(_frame(0), index * 0.1).process is True

    def test_describe_names_the_zone_when_active(self):
        assert "zone=0,0,200,400" in _gate().describe()


# ---------------------------------------------------------------------------
# The decision itself
# ---------------------------------------------------------------------------

class TestGateDecision:
    def test_warmup_frames_are_always_processed(self):
        gate = _gate(_settings(warmup_frames=5))
        reasons = [gate.should_process(_frame(0), i * 0.1).reason for i in range(5)]
        assert reasons == ["warmup"] * 5

    def test_static_scene_is_gated_once_warm(self):
        decision, _ = _settle(_gate())
        assert decision.process is False
        assert decision.reason == "idle"

    def test_motion_in_the_entry_zone_wakes_detection(self):
        gate = _gate()
        _, ts = _settle(gate)
        decision = gate.should_process(_frame_with_blob(x=100), ts)
        assert decision.process is True
        assert decision.reason == "motion"

    def test_change_fraction_is_reported(self):
        gate = _gate()
        _, ts = _settle(gate)
        decision = gate.should_process(_frame_with_blob(x=100), ts)
        assert decision.change_fraction > 0.0

    def test_motion_outside_the_entry_zone_is_ignored(self):
        # A bike near the far edge is past the entry zone; the gate must rely
        # on note_detection for that case, not on seeing it here.
        gate = _gate()
        _, ts = _settle(gate)
        decision = gate.should_process(_frame_with_blob(x=700), ts)
        assert decision.process is False

    def test_change_below_the_threshold_stays_gated(self):
        gate = _gate(_settings(min_area_fraction=0.5))
        _, ts = _settle(gate)
        # A blob covering far less than half the strip.
        assert gate.should_process(_frame_with_blob(x=100, size=20), ts).process is False

    def test_a_lower_threshold_makes_the_same_change_trigger(self):
        gate = _gate(_settings(min_area_fraction=0.0005))
        _, ts = _settle(gate)
        assert gate.should_process(_frame_with_blob(x=100, size=20), ts).process is True

    def test_pixel_delta_ignores_faint_changes(self):
        gate = _gate(_settings(pixel_delta=200.0))
        _, ts = _settle(gate)
        # A dim blob: present, but nowhere near 200 grey levels of change.
        assert gate.should_process(_frame_with_blob(x=100, value=30), ts).process is False

    def test_zone_fraction_one_watches_the_whole_frame(self):
        gate = _gate(_settings(zone_fraction=1.0))
        _, ts = _settle(gate)
        assert gate.should_process(_frame_with_blob(x=700), ts).process is True


# ---------------------------------------------------------------------------
# The hold — the safety net for riders already past the entry zone
# ---------------------------------------------------------------------------

class TestGateHold:
    def test_a_noted_detection_keeps_the_next_frame_awake(self):
        gate = _gate(_settings(hold_sec=1.0))
        _, ts = _settle(gate)
        gate.note_detection(ts)
        decision = gate.should_process(_frame(0), ts + 0.1)
        assert decision.process is True
        assert decision.reason == "hold"

    def test_the_hold_expires(self):
        gate = _gate(_settings(hold_sec=1.0))
        _, ts = _settle(gate)
        gate.note_detection(ts)
        assert gate.should_process(_frame(0), ts + 1.5).process is False

    def test_the_hold_holds_right_up_to_its_limit(self):
        gate = _gate(_settings(hold_sec=1.0))
        _, ts = _settle(gate)
        gate.note_detection(ts)
        assert gate.should_process(_frame(0), ts + 1.0).process is True

    def test_repeated_detections_renew_the_hold_indefinitely(self):
        gate = _gate(_settings(hold_sec=0.5))
        _, ts = _settle(gate)
        for step in range(30):
            frame_ts = ts + step * 0.1
            gate.note_detection(frame_ts)
            assert gate.should_process(_frame(0), frame_ts + 0.05).process is True

    def test_motion_itself_starts_a_hold(self):
        gate = _gate(_settings(hold_sec=1.0))
        _, ts = _settle(gate)
        gate.should_process(_frame_with_blob(x=100), ts)
        # The blob is gone again, but the hold from the motion still covers us.
        assert gate.should_process(_frame(0), ts + 0.2).reason == "hold"

    def test_zero_hold_gates_immediately_after_the_detection_frame(self):
        gate = _gate(_settings(hold_sec=0.0))
        _, ts = _settle(gate)
        gate.note_detection(ts)
        assert gate.should_process(_frame(0), ts + 0.1).process is False


# ---------------------------------------------------------------------------
# Background adaptation
# ---------------------------------------------------------------------------

class TestBackgroundAdaptation:
    def test_a_new_static_scene_is_eventually_absorbed(self):
        # A parked bike, or the sun coming out, must not hold the gate open
        # forever once the scene has stopped changing.
        gate = _gate(_settings(background_alpha=0.5, hold_sec=0.0))
        _, ts = _settle(gate)
        frame = _frame_with_blob(x=100)
        reasons = []
        for step in range(40):
            reasons.append(gate.should_process(frame, ts + step * 0.1).reason)
        assert reasons[0] == "motion"
        assert reasons[-1] == "idle", reasons

    def test_a_slow_lighting_drift_does_not_wake_detection(self):
        gate = _gate(_settings(background_alpha=0.5, hold_sec=0.0))
        _settle(gate)
        awake = 0
        for step in range(40):
            # One grey level every frame — far below pixel_delta.
            if gate.should_process(_frame(step // 4), 1.0 + step * 0.1).process:
                awake += 1
        assert awake == 0

    def test_the_first_frame_is_processed_with_no_background_to_compare(self):
        gate = _gate(_settings(warmup_frames=0))
        decision = gate.should_process(_frame_with_blob(x=100), 0.0)
        assert decision.process is True
        assert decision.change_fraction == 1.0


# ---------------------------------------------------------------------------
# Reported statistics
# ---------------------------------------------------------------------------

def stats_fraction(gate):
    return gate.stats()["gated_fraction"]


class TestGateStats:
    def test_counts_add_up_to_the_frames_seen(self):
        gate = _gate()
        for step in range(20):
            gate.should_process(_frame(0), step * 0.1)
        stats = gate.stats()
        assert stats["processed_frames"] + stats["gated_frames"] == 20

    def test_gated_fraction_is_reported(self):
        # Frame 1 initialises the background and is processed; the nine static
        # frames after it are skipped.
        gate = _gate(_settings(warmup_frames=0, hold_sec=0.0))
        for step in range(10):
            gate.should_process(_frame(0), step * 0.1)
        assert stats_fraction(gate) == pytest.approx(0.9, abs=0.01)

    def test_disabled_gate_reports_zero_gated(self):
        gate = _gate(_settings(enabled=False))
        for step in range(10):
            gate.should_process(_frame(0), step * 0.1)
        stats = gate.stats()
        assert stats["gated_frames"] == 0
        assert stats["enabled"] is False

    def test_stats_include_the_zone(self):
        assert _gate().stats()["zone"] == [0, 0, 200, 400]

    def test_stats_explain_why_a_gate_is_off(self):
        gate = _gate(line_a=(-100, 0), line_b=(-100, 400))
        assert "ambiguous" in gate.stats()["disabled_reason"]

    def test_empty_run_reports_a_zero_fraction(self):
        assert _gate().stats()["gated_fraction"] == 0.0


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------

class TestGateRobustness:
    def test_grayscale_frames_are_accepted(self):
        gate = _gate()
        for step in range(6):
            decision = gate.should_process(np.zeros((FRAME_H, FRAME_W), np.uint8), step * 0.1)
        assert isinstance(decision, GateDecision)

    def test_downscale_one_analyses_full_resolution(self):
        gate = _gate(_settings(downscale=1))
        _, ts = _settle(gate)
        assert gate.should_process(_frame_with_blob(x=100), ts).process is True

    def test_a_strip_narrower_than_the_downscale_still_works(self):
        gate = _gate(_settings(downscale=64, zone_fraction=0.01))
        for step in range(6):
            decision = gate.should_process(_frame(0), step * 0.1)
        assert isinstance(decision, GateDecision)

    def test_a_frame_smaller_than_the_zone_does_not_raise(self):
        gate = _gate()
        # A source that changed resolution mid-run: the crop comes out short.
        decision = gate.should_process(np.zeros((10, 10, 3), np.uint8), 0.0)
        assert decision.process is True
