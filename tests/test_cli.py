"""Tests for cli.py — argument parsing and how flags override the config file.

Every detection subcommand accepts the same settings from two places: a YAML
config and command-line flags. These tests pin down the precedence and the
config-file fallbacks, and they stub the pipeline entry points so nothing here
loads a model or opens a video.
"""
import json

import pytest
import yaml

from mx_tracker import cli


@pytest.fixture
def tracker_config(tmp_path):
    """A minimal tracker config that also carries source/out_dir defaults."""
    path = tmp_path / "run.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "source": "from_config.mp4",
                "out_dir": "artifacts/from_config",
                "line": {"direction": "right_to_left", "width": 40},
                "models": {"vehicle_model": "config_vehicle.pt", "plate_zone_n": 4,
                           "plate_zone_select": [1, 2]},
                "reid": {"enabled": True, "gallery_path": "config_gallery"},
            }
        )
    )
    return path


@pytest.fixture
def captured(monkeypatch):
    """Replace the pipeline entry points with recorders."""
    calls: dict[str, dict] = {}

    def recorder(name):
        def fake(**kwargs):
            calls[name] = kwargs
            return {"ok": True, "run_dir": "recorded"}

        return fake

    for name in ("run_file_detection", "run_stream_detection", "collect_samples"):
        monkeypatch.setattr(cli, name, recorder(name))
    return calls


def _run(argv):
    return cli.main(argv)


# ---------------------------------------------------------------------------
# Parser wiring
# ---------------------------------------------------------------------------

class TestParser:
    def test_command_is_required(self):
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args([])

    def test_detect_requires_a_subcommand(self):
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(["detect"])

    @pytest.mark.parametrize(
        "argv",
        [
            ["detect", "file", "--source", "a.mp4"],
            ["detect", "stream", "--source", "a.mp4"],
            ["collect", "--source", "a.mp4"],
            ["dataset", "validate", "--dataset-dir", "d"],
            ["recount", "--run-dir", "r"],
            ["reid-watch", "--run-dir", "r"],
            ["serve"],
            ["config", "init"],
        ],
    )
    def test_every_subcommand_binds_a_handler(self, argv):
        assert callable(cli.build_parser().parse_args(argv).func)

    def test_collect_defaults_to_file_mode(self):
        args = cli.build_parser().parse_args(["collect", "--source", "a.mp4"])
        assert args.source_mode == "file"

    def test_collect_rejects_an_unknown_source_mode(self):
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(["collect", "--source", "a.mp4", "--source-mode", "udp"])


# ---------------------------------------------------------------------------
# source / out_dir fallback to the config file
#
# Regression: `collect` read args.source directly, so `--config` alone crashed
# with an AttributeError deep inside VideoSource instead of using the config's
# source (or reporting a clear error when there was none).
# ---------------------------------------------------------------------------

class TestSourceAndOutDirFallback:
    @pytest.mark.parametrize(
        "argv,recorded",
        [
            (["detect", "file"], "run_file_detection"),
            (["detect", "stream"], "run_stream_detection"),
            (["collect"], "collect_samples"),
        ],
    )
    def test_source_comes_from_the_config_when_no_flag_is_given(
        self, argv, recorded, tracker_config, captured, capsys
    ):
        _run(argv + ["--config", str(tracker_config)])
        assert captured[recorded]["source"] == "from_config.mp4"

    @pytest.mark.parametrize(
        "argv,recorded",
        [
            (["detect", "file"], "run_file_detection"),
            (["detect", "stream"], "run_stream_detection"),
            (["collect"], "collect_samples"),
        ],
    )
    def test_out_dir_comes_from_the_config_when_no_flag_is_given(
        self, argv, recorded, tracker_config, captured, capsys
    ):
        _run(argv + ["--config", str(tracker_config)])
        assert captured[recorded]["output_dir"] == "artifacts/from_config"

    @pytest.mark.parametrize(
        "argv,recorded",
        [
            (["detect", "file"], "run_file_detection"),
            (["detect", "stream"], "run_stream_detection"),
            (["collect"], "collect_samples"),
        ],
    )
    def test_flags_win_over_the_config(self, argv, recorded, tracker_config, captured, capsys):
        _run(argv + ["--config", str(tracker_config), "--source", "cli.mp4", "--out-dir", "cli_out"])
        assert captured[recorded]["source"] == "cli.mp4"
        assert captured[recorded]["output_dir"] == "cli_out"

    @pytest.mark.parametrize(
        "argv", [["detect", "file"], ["detect", "stream"], ["collect"]]
    )
    def test_missing_source_everywhere_reports_a_clear_error(self, argv, captured):
        with pytest.raises(SystemExit, match="--source is required"):
            _run(argv)

    @pytest.mark.parametrize(
        "argv", [["detect", "file"], ["detect", "stream"], ["collect"]]
    )
    def test_config_without_a_source_key_reports_a_clear_error(self, argv, tmp_path, captured):
        config = tmp_path / "no_source.yaml"
        config.write_text(yaml.safe_dump({"line": {"width": 10}}))
        with pytest.raises(SystemExit, match="--source is required"):
            _run(argv + ["--config", str(config)])


# ---------------------------------------------------------------------------
# Settings overrides
# ---------------------------------------------------------------------------

class TestSettingsOverrides:
    def _settings(self, argv, captured, recorded="run_file_detection"):
        _run(argv)
        return captured[recorded]["settings"]

    def test_config_values_reach_the_pipeline(self, tracker_config, captured, capsys):
        settings = self._settings(
            ["detect", "file", "--config", str(tracker_config), "--source", "a.mp4"], captured
        )
        assert settings.line.direction == "right_to_left"
        assert settings.line.width == 40
        assert settings.models.vehicle_model == "config_vehicle.pt"

    def test_unspecified_sections_keep_their_defaults(self, tracker_config, captured, capsys):
        settings = self._settings(
            ["detect", "file", "--config", str(tracker_config), "--source", "a.mp4"], captured
        )
        assert settings.reads.vote_window_sec == pytest.approx(2.5)

    def test_line_and_width_flags_override_the_config(self, tracker_config, captured, capsys):
        settings = self._settings(
            ["detect", "file", "--config", str(tracker_config), "--source", "a.mp4",
             "--line", "10%,0%,10%,100%", "--line-width", "8"],
            captured,
        )
        assert settings.line.value == "10%,0%,10%,100%"
        assert settings.line.width == 8

    def test_model_and_device_flags_override_the_config(self, tracker_config, captured, capsys):
        settings = self._settings(
            ["detect", "file", "--config", str(tracker_config), "--source", "a.mp4",
             "--vehicle-model", "v.pt", "--plate-model", "p.pt", "--device", "cpu"],
            captured,
        )
        assert (settings.models.vehicle_model, settings.models.plate_model) == ("v.pt", "p.pt")
        assert settings.runtime.device == "cpu"

    def test_disable_reid_overrides_an_enabled_config(self, tracker_config, captured, capsys):
        settings = self._settings(
            ["detect", "file", "--config", str(tracker_config), "--source", "a.mp4", "--disable-reid"],
            captured,
        )
        assert settings.reid.enabled is False

    def test_enable_reid_and_gallery_flags(self, captured, capsys):
        settings = self._settings(
            ["detect", "file", "--source", "a.mp4", "--enable-reid", "--gallery", "g"], captured
        )
        assert settings.reid.enabled is True
        assert settings.reid.gallery_path == "g"

    def test_digits_only_drops_the_plate_class(self, captured, capsys):
        settings = self._settings(
            ["detect", "file", "--source", "a.mp4", "--digits-only"], captured
        )
        assert settings.models.plate_has_class is False

    def test_stream_flags_reach_the_stream_settings(self, captured, capsys):
        _run(["detect", "stream", "--source", "a.mp4", "--loop-source",
              "--reconnect-delay", "1.5", "--max-reconnects", "4"])
        settings = captured["run_stream_detection"]["settings"]
        assert settings.stream.loop_file is True
        assert settings.stream.reconnect_delay_sec == pytest.approx(1.5)
        assert settings.stream.max_reconnects == 4

    def test_limit_frames_is_passed_through(self, captured, capsys):
        _run(["detect", "file", "--source", "a.mp4", "--limit-frames", "50"])
        assert captured["run_file_detection"]["limit_frames"] == 50

    def test_limit_frames_defaults_to_none(self, captured, capsys):
        _run(["detect", "file", "--source", "a.mp4"])
        assert captured["run_file_detection"]["limit_frames"] is None

    def test_collect_source_mode_reaches_the_pipeline(self, captured, capsys):
        _run(["collect", "--source", "a.mp4", "--source-mode", "stream"])
        assert captured["collect_samples"]["mode"] == "stream"

    def test_base_dir_is_the_config_directory(self, tracker_config, captured, capsys):
        _run(["detect", "file", "--config", str(tracker_config), "--source", "a.mp4"])
        assert captured["run_file_detection"]["base_dir"] == tracker_config.parent


# ---------------------------------------------------------------------------
# Handler output and per-script configs
# ---------------------------------------------------------------------------

class TestHandlerOutput:
    def test_detect_prints_the_summary_as_json(self, captured, capsys):
        _run(["detect", "file", "--source", "a.mp4"])
        assert json.loads(capsys.readouterr().out)["run_dir"] == "recorded"

    def test_handlers_return_zero_on_success(self, captured, capsys):
        assert _run(["detect", "file", "--source", "a.mp4"]) == 0

    def test_config_init_writes_a_loadable_config(self, tmp_path, capsys):
        destination = tmp_path / "generated.yaml"
        assert _run(["config", "init", "--output", str(destination)]) == 0
        from mx_tracker.config import load_settings

        settings, _ = load_settings(destination)
        assert settings.line.direction in {"left_to_right", "right_to_left"}


class TestScriptConfigLoading:
    def test_missing_optional_config_is_ignored(self, tmp_path):
        assert cli._load_script_config(str(tmp_path / "absent.yaml")) == {}

    def test_missing_required_config_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            cli._load_script_config(str(tmp_path / "absent.yaml"), required=True)

    def test_no_config_path_is_ignored(self):
        assert cli._load_script_config(None) == {}

    def test_empty_config_file_reads_as_empty_mapping(self, tmp_path):
        path = tmp_path / "empty.yaml"
        path.write_text("")
        assert cli._load_script_config(str(path)) == {}

    def test_values_are_returned_as_parsed(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text(yaml.safe_dump({"run_dir": "r", "threshold": 0.5}))
        assert cli._load_script_config(str(path)) == {"run_dir": "r", "threshold": 0.5}


class TestRecountHandler:
    def test_run_dir_falls_back_to_the_config(self, tmp_path, monkeypatch):
        recorded: dict = {}
        monkeypatch.setattr(cli, "recount", lambda **kw: recorded.update(kw))
        config = tmp_path / "recount.yaml"
        config.write_text(yaml.safe_dump({"run_dir": "runs/one", "race_start_sec": 12.0}))
        _run(["recount", "--config", str(config)])
        assert recorded["run_dir"] == "runs/one"
        assert recorded["race_start_sec"] == pytest.approx(12.0)

    def test_flags_override_the_config(self, tmp_path, monkeypatch):
        recorded: dict = {}
        monkeypatch.setattr(cli, "recount", lambda **kw: recorded.update(kw))
        config = tmp_path / "recount.yaml"
        config.write_text(yaml.safe_dump({"run_dir": "runs/one", "race_start_sec": 12.0}))
        _run(["recount", "--config", str(config), "--run-dir", "runs/two", "--race-start-sec", "3"])
        assert recorded["run_dir"] == "runs/two"
        assert recorded["race_start_sec"] == pytest.approx(3.0)

    def test_race_start_at_is_passed_through(self, tmp_path, monkeypatch):
        recorded: dict = {}
        monkeypatch.setattr(cli, "recount", lambda **kw: recorded.update(kw))
        _run(["recount", "--run-dir", "r", "--race-start-at", "10:31:00"])
        assert recorded["race_start_at"] == "10:31:00"

    def test_missing_run_dir_reports_a_clear_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cli, "recount", lambda **kw: None)
        empty = tmp_path / "empty.yaml"
        empty.write_text("")
        with pytest.raises(SystemExit, match="--run-dir is required"):
            _run(["recount", "--config", str(empty)])
