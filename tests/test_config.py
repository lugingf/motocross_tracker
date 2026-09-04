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


# ---------------------------------------------------------------------------
# Defaults must satisfy their own validators.
#
# Pydantic skips validation of field defaults, so an invalid default is silent
# until it reaches the crossing logic — where an unrecognised direction filters
# out every crossing and the run finds nothing.
# ---------------------------------------------------------------------------

class TestDefaultsAreSelfConsistent:
    def test_line_direction_default_passes_its_own_validator(self):
        assert TrackerSettings().line.direction in {"left_to_right", "right_to_left"}

    def test_line_settings_default_matches_default_config(self):
        from mx_tracker.config import DEFAULT_CONFIG, LineSettings

        assert LineSettings().direction == DEFAULT_CONFIG["line"]["direction"]

    def test_every_default_config_section_round_trips(self):
        from mx_tracker.config import DEFAULT_CONFIG

        settings, _ = load_settings(None)
        for section in DEFAULT_CONFIG:
            assert hasattr(settings, section)


# ---------------------------------------------------------------------------
# plate_zone_select must address zones that actually exist — an out-of-range
# entry used to silently fall back to the full frame instead of erroring.
# ---------------------------------------------------------------------------

class TestPlateZoneSelectValidation:
    def _models(self, n, select):
        from mx_tracker.config import ModelSettings

        return ModelSettings(plate_zone_n=n, plate_zone_select=select)

    def test_valid_selection_accepted(self):
        assert self._models(9, [4, 5, 7, 8]).plate_zone_select == [4, 5, 7, 8]

    def test_full_selection_accepted(self):
        assert self._models(4, [1, 2, 3, 4]).plate_zone_select == [1, 2, 3, 4]

    def test_empty_selection_accepted_as_full_frame(self):
        assert self._models(9, []).plate_zone_select == []

    def test_zone_above_grid_size_rejected(self):
        with pytest.raises(ValidationError):
            self._models(4, [5])

    def test_zone_zero_rejected(self):
        with pytest.raises(ValidationError):
            self._models(4, [0])

    def test_negative_zone_rejected(self):
        with pytest.raises(ValidationError):
            self._models(4, [-1])

    def test_error_names_the_offending_zone(self):
        with pytest.raises(ValidationError, match="99"):
            self._models(9, [1, 99])

    def test_invalid_selection_in_yaml_raises(self, tmp_path):
        config = tmp_path / "bad.yaml"
        config.write_text("models:\n  plate_zone_n: 4\n  plate_zone_select: [7]\n")
        with pytest.raises(ValidationError):
            load_settings(config)

    def test_shipped_configs_are_valid(self):
        from mx_tracker.runtime import REPO_ROOT

        configs = sorted((REPO_ROOT / "configs").glob("*.yaml"))
        assert configs, "expected shipped configs to exist"
        for path in configs:
            # Per-script configs (train/recount/reid_watch) are not tracker
            # configs; load_settings must still tolerate their extra keys.
            load_settings(path)
