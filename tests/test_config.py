"""Tests for config.py — settings validation and loading."""
import yaml
import pytest
from pathlib import Path
from pydantic import ValidationError

from mx_tracker.config import LineSettings, TrackerSettings, load_settings, resolve_path, write_default_config


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


class TestResolvePath:
    def test_none_returns_none(self, tmp_path):
        assert resolve_path(tmp_path, None) is None

    def test_empty_string_returns_none(self, tmp_path):
        assert resolve_path(tmp_path, "") is None

    def test_absolute_path_returned_as_path_object(self, tmp_path):
        result = resolve_path(tmp_path, "/some/absolute/file.txt")
        assert result == Path("/some/absolute/file.txt")

    def test_relative_path_resolved_against_base_dir(self, tmp_path):
        result = resolve_path(tmp_path, "subdir/file.txt")
        assert result == (tmp_path / "subdir" / "file.txt").resolve()


class TestWriteDefaultConfig:
    def test_creates_yaml_file_at_destination(self, tmp_path):
        out = tmp_path / "config.yaml"
        write_default_config(out)
        assert out.exists()

    def test_file_contains_valid_yaml(self, tmp_path):
        out = tmp_path / "config.yaml"
        write_default_config(out)
        data = yaml.safe_load(out.read_text())
        assert isinstance(data, dict)

    def test_written_config_is_loadable_by_load_settings(self, tmp_path):
        out = tmp_path / "default.yaml"
        write_default_config(out)
        settings, _ = load_settings(out)
        assert isinstance(settings, TrackerSettings)

    def test_creates_parent_directories(self, tmp_path):
        out = tmp_path / "deep" / "nested" / "config.yaml"
        write_default_config(out)
        assert out.exists()

    def test_returns_resolved_path(self, tmp_path):
        out = tmp_path / "cfg.yaml"
        result = write_default_config(out)
        assert result == out.resolve()


class TestLoadSettingsWithYaml:
    def test_yaml_overrides_default_direction(self, tmp_path):
        cfg = tmp_path / "my.yaml"
        cfg.write_text("line:\n  direction: right_to_left\n")
        settings, _ = load_settings(cfg)
        assert settings.line.direction == "right_to_left"

    def test_base_dir_is_config_file_parent(self, tmp_path):
        subdir = tmp_path / "conf"
        subdir.mkdir()
        cfg = subdir / "settings.yaml"
        cfg.write_text("{}")
        _, base_dir = load_settings(cfg)
        assert base_dir == subdir

    def test_partial_yaml_preserves_other_defaults(self, tmp_path):
        cfg = tmp_path / "partial.yaml"
        cfg.write_text("reads:\n  min_digits: 3\n")
        settings, _ = load_settings(cfg)
        assert settings.reads.min_digits == 3
        assert settings.reads.vote_window_sec == pytest.approx(2.5)

    def test_invalid_direction_in_yaml_raises_validation_error(self, tmp_path):
        cfg = tmp_path / "bad.yaml"
        cfg.write_text("line:\n  direction: both\n")
        with pytest.raises(Exception):
            load_settings(cfg)
