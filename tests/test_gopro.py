"""Tests for gopro.py — file discovery, concat list building, ffmpeg command construction.

No subprocess calls are made: _build_ffmpeg_transcode_command is a pure function,
and prepare_gopro_video (which calls subprocess) is not exercised here.
"""
from pathlib import Path

import pytest

from mx_tracker.gopro import (
    _build_ffmpeg_transcode_command,
    _parse_gopro_name,
    build_concat_list,
    discover_gopro_files,
)


# ---------------------------------------------------------------------------
# _parse_gopro_name — regex parsing of stem into (prefix, chapter, clip)
# ---------------------------------------------------------------------------

class TestParseGoProName:
    def test_valid_name_returns_gopro_file(self):
        result = _parse_gopro_name(Path("/d/GX010001.mp4"))
        assert result is not None
        assert result.prefix == "GX"
        assert result.chapter == 1
        assert result.clip == 1

    def test_prefix_is_uppercased(self):
        result = _parse_gopro_name(Path("/d/gx010001.mp4"))
        assert result is not None
        assert result.prefix == "GX"

    def test_chapter_and_clip_extracted_correctly(self):
        result = _parse_gopro_name(Path("/d/GH023456.mp4"))
        assert result is not None
        assert result.chapter == 2
        assert result.clip == 3456

    def test_non_gopro_name_returns_none(self):
        assert _parse_gopro_name(Path("/d/video.mp4")) is None

    def test_too_few_digits_returns_none(self):
        # Pattern needs exactly 2+2+4 chars; GX01001 has only 7
        assert _parse_gopro_name(Path("/d/GX01001.mp4")) is None

    def test_numeric_prefix_returns_none(self):
        assert _parse_gopro_name(Path("/d/12010001.mp4")) is None

    def test_three_letter_prefix_returns_none(self):
        # [A-Za-z]{2} requires exactly 2 letters
        assert _parse_gopro_name(Path("/d/GXY10001.mp4")) is None


# ---------------------------------------------------------------------------
# discover_gopro_files — filesystem glob + filtering + sorting
# ---------------------------------------------------------------------------

class TestDiscoverGoProFiles:
    def test_finds_gopro_mp4_files(self, tmp_path):
        (tmp_path / "GX010001.mp4").touch()
        (tmp_path / "GX010002.mp4").touch()
        result = discover_gopro_files(str(tmp_path / "*.mp4"))
        assert len(result) == 2

    def test_ignores_non_gopro_names(self, tmp_path):
        (tmp_path / "GX010001.mp4").touch()
        (tmp_path / "video.mp4").touch()
        result = discover_gopro_files(str(tmp_path / "*.mp4"))
        assert len(result) == 1
        assert result[0].prefix == "GX"

    def test_ignores_non_video_extensions(self, tmp_path):
        (tmp_path / "GX010001.mp4").touch()
        (tmp_path / "GX010002.txt").touch()
        result = discover_gopro_files(str(tmp_path / "GX01000*"))
        assert len(result) == 1

    def test_returns_empty_list_when_no_files_match(self, tmp_path):
        result = discover_gopro_files(str(tmp_path / "*.mp4"))
        assert result == []

    def test_sorted_by_clip_first(self, tmp_path):
        # clip 0002 must come before clip 0003 regardless of chapter
        (tmp_path / "GX020003.mp4").touch()
        (tmp_path / "GX010002.mp4").touch()
        result = discover_gopro_files(str(tmp_path / "*.mp4"))
        assert result[0].clip == 2
        assert result[1].clip == 3

    def test_directories_named_like_gopro_files_ignored(self, tmp_path):
        gopro_dir = tmp_path / "GX010001.mp4"
        gopro_dir.mkdir()
        result = discover_gopro_files(str(tmp_path / "*.mp4"))
        assert len(result) == 0

    def test_mov_extension_accepted(self, tmp_path):
        (tmp_path / "GX010001.MOV").touch()
        result = discover_gopro_files(str(tmp_path / "GX010001.MOV"))
        assert len(result) == 1


# ---------------------------------------------------------------------------
# build_concat_list — creates ffmpeg concat file from discovered GoPro files
# ---------------------------------------------------------------------------

class TestBuildConcatList:
    def test_creates_concat_file_with_file_entries(self, tmp_path):
        (tmp_path / "GX010001.mp4").touch()
        (tmp_path / "GX010002.mp4").touch()
        output = tmp_path / "concat.txt"
        build_concat_list(str(tmp_path / "*.mp4"), output)
        content = output.read_text()
        assert "GX010001.mp4" in content
        assert "GX010002.mp4" in content

    def test_each_entry_starts_with_file_keyword(self, tmp_path):
        (tmp_path / "GX010001.mp4").touch()
        output = tmp_path / "list.txt"
        build_concat_list(str(tmp_path / "*.mp4"), output)
        for line in output.read_text().splitlines():
            assert line.startswith("file ")

    def test_report_lists_resolved_file_paths(self, tmp_path):
        (tmp_path / "GX010001.mp4").touch()
        output = tmp_path / "list.txt"
        report = build_concat_list(str(tmp_path / "*.mp4"), output)
        assert len(report.files) == 1

    def test_report_output_list_is_absolute(self, tmp_path):
        output = tmp_path / "list.txt"
        report = build_concat_list(str(tmp_path / "*.mp4"), output)
        assert report.output_list == str(output.resolve())

    def test_empty_pattern_creates_empty_file(self, tmp_path):
        output = tmp_path / "concat.txt"
        build_concat_list(str(tmp_path / "*.mp4"), output)
        assert output.read_text() == ""

    def test_creates_parent_directories(self, tmp_path):
        output = tmp_path / "deep" / "nested" / "list.txt"
        build_concat_list(str(tmp_path / "*.mp4"), output)
        assert output.exists()


# ---------------------------------------------------------------------------
# _build_ffmpeg_transcode_command — pure function, builds a shell command list
# ---------------------------------------------------------------------------

class TestBuildFfmpegTranscodeCommand:
    def _cmd(self, **overrides):
        defaults = dict(
            ffmpeg_bin="ffmpeg",
            input_path=Path("in.mp4"),
            output_path=Path("out.mp4"),
            crf=18,
            preset="veryfast",
            video_codec="libx264",
            audio_codec="copy",
            fps=None,
            scale=None,
        )
        defaults.update(overrides)
        return _build_ffmpeg_transcode_command(**defaults)

    def test_first_element_is_ffmpeg_bin(self):
        assert self._cmd(ffmpeg_bin="ffmpeg_custom")[0] == "ffmpeg_custom"

    def test_last_element_is_output_path(self):
        assert self._cmd(output_path=Path("result.mp4"))[-1] == "result.mp4"

    def test_input_flag_followed_by_input_path(self):
        cmd = self._cmd(input_path=Path("source.mp4"))
        idx = cmd.index("-i")
        assert cmd[idx + 1] == "source.mp4"

    def test_crf_value_in_command(self):
        cmd = self._cmd(crf=23)
        idx = cmd.index("-crf")
        assert cmd[idx + 1] == "23"

    def test_preset_value_in_command(self):
        cmd = self._cmd(preset="slow")
        idx = cmd.index("-preset")
        assert cmd[idx + 1] == "slow"

    def test_fps_flag_added_when_specified(self):
        cmd = self._cmd(fps=60)
        assert "-r" in cmd
        assert cmd[cmd.index("-r") + 1] == "60"

    def test_fps_flag_absent_when_none(self):
        assert "-r" not in self._cmd(fps=None)

    def test_scale_flag_added_when_specified(self):
        cmd = self._cmd(scale="1280:-1")
        assert "-vf" in cmd
        assert "scale=1280:-1" in cmd[cmd.index("-vf") + 1]

    def test_scale_flag_absent_when_none(self):
        assert "-vf" not in self._cmd(scale=None)

    def test_video_codec_in_command(self):
        cmd = self._cmd(video_codec="libx265")
        assert "libx265" in cmd

    def test_audio_codec_in_command(self):
        cmd = self._cmd(audio_codec="aac")
        assert "aac" in cmd

    def test_overwrite_flag_present(self):
        assert "-y" in self._cmd()
