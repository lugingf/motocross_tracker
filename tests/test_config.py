"""Tests for config.py — settings validation and loading."""
import pytest
from pydantic import ValidationError

from mx_tracker.config import LineSettings, TrackerSettings, load_settings


class TestDirectionValidation:
    def test_left_to_right_accepted(self):
        s = LineSettings(direction="left_to_right")
        assert s.direction == "left_to_right"

    def test_right_to_left_accepted(self):
        s = LineSettings(direction="right_to_left")
        assert s.direction == "right_to_left"

    def test_either_rejected(self):
        with pytest.raises(ValidationError):
            LineSettings(direction="either")

    def test_empty_string_rejected(self):
        with pytest.raises(ValidationError):
            LineSettings(direction="")

    def test_positive_negative_rejected(self):
        with pytest.raises(ValidationError):
            LineSettings(direction="positive")
        with pytest.raises(ValidationError):
            LineSettings(direction="negative")


class TestDefaultSettings:
    def test_load_settings_without_config_returns_defaults(self):
        settings, _ = load_settings(None)
        assert isinstance(settings, TrackerSettings)
        assert settings.line.direction == "left_to_right"
        assert settings.models.plate_has_class is True
        assert settings.reads.min_digits == 1

    def test_plate_zone_n_must_be_supported_value(self):
        from mx_tracker.config import ModelSettings
        with pytest.raises(ValidationError):
            ModelSettings(plate_zone_n=3)   # 3 is not in {1, 2, 4, 9, 32}

    def test_supported_plate_zone_values(self):
        from mx_tracker.config import ModelSettings
        for n in (1, 2, 4, 9, 32):
            m = ModelSettings(plate_zone_n=n)
            assert m.plate_zone_n == n
