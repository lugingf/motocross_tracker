from __future__ import annotations

import glob
import re
import shlex
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


GOPRO_PATTERN = re.compile(r"^(?P<prefix>[A-Za-z]{2})(?P<chapter>\d{2})(?P<clip>\d{4})$")
VIDEO_EXTENSIONS = {".mp4", ".mov", ".MP4", ".MOV"}


@dataclass(slots=True)
class GoProFile:
    path: Path
    prefix: str
    chapter: int
    clip: int


@dataclass(slots=True)
class GoProListReport:
    pattern: str
    output_list: str
    files: list[str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class GoProPrepareReport:
    concat_list: str | None
    merged_output: str | None
    transcoded_output: str
    files: list[str]
    ffmpeg_command: list[str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _parse_gopro_name(path: Path) -> GoProFile | None:
    match = GOPRO_PATTERN.match(path.stem)
    if match is None:
        return None
    return GoProFile(
        path=path.resolve(),
        prefix=match.group("prefix").upper(),
        chapter=int(match.group("chapter")),
        clip=int(match.group("clip")),
    )


def discover_gopro_files(pattern: str, recursive: bool = False) -> list[GoProFile]:
    matched_paths = [Path(value).expanduser() for value in glob.glob(pattern, recursive=recursive)]
    parsed: list[GoProFile] = []
    for path in matched_paths:
        if path.suffix not in VIDEO_EXTENSIONS or not path.is_file():
            continue
        parsed_item = _parse_gopro_name(path)
        if parsed_item is None:
            continue
        parsed.append(parsed_item)
    parsed.sort(key=lambda item: (item.clip, item.chapter, item.prefix, str(item.path)))
    return parsed


def build_concat_list(
    pattern: str,
    output_list: str | Path,
    recursive: bool = False,
) -> GoProListReport:
    files = discover_gopro_files(pattern, recursive=recursive)
    output_path = Path(output_list).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(f"file {shlex.quote(str(item.path))}\n" for item in files))
    return GoProListReport(
        pattern=pattern,
        output_list=str(output_path),
        files=[str(item.path) for item in files],
    )


def _run_command(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _build_ffmpeg_transcode_command(
    ffmpeg_bin: str,
    input_path: Path,
    output_path: Path,
    crf: int,
    preset: str,
    video_codec: str,
    audio_codec: str,
    fps: int | None = None,
    scale: str | None = None,
) -> list[str]:
    command = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(input_path),
        "-c:v",
        video_codec,
        "-crf",
        str(crf),
        "-preset",
        preset,
    ]
    if fps is not None:
        command.extend(["-r", str(fps)])
    if scale:
        command.extend(["-vf", f"scale={scale}"])
    command.extend(
        [
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-c:a",
            audio_codec,
            str(output_path),
        ]
    )
    return command


def prepare_gopro_video(
    pattern: str,
    output_dir: str | Path,
    name: str,
    ffmpeg_bin: str = "ffmpeg",
    crf: int = 18,
    preset: str = "veryfast",
    video_codec: str = "libx264",
    audio_codec: str = "copy",
    fps: int | None = None,
    scale: str | None = None,
    recursive: bool = False,
    transcode: bool = True,
) -> GoProPrepareReport:
    files = discover_gopro_files(pattern, recursive=recursive)
    if not files:
        raise FileNotFoundError(f"No GoPro files found for pattern: {pattern}")

    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    concat_list_path = out_dir / f"{name}_concat.txt"
    merge_path = out_dir / f"{name}_merged.mp4"
    final_path = out_dir / (f"{name}_merged_h264.mp4" if transcode else f"{name}_merged.mp4")

    build_concat_list(pattern=pattern, output_list=concat_list_path, recursive=recursive)
    _run_command(
        [
            ffmpeg_bin,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list_path),
            "-c",
            "copy",
            str(merge_path),
        ]
    )

    ffmpeg_command: list[str]
    if transcode:
        ffmpeg_command = _build_ffmpeg_transcode_command(
            ffmpeg_bin=ffmpeg_bin,
            input_path=merge_path,
            output_path=final_path,
            crf=crf,
            preset=preset,
            video_codec=video_codec,
            audio_codec=audio_codec,
            fps=fps,
            scale=scale,
        )
        _run_command(ffmpeg_command)
    else:
        ffmpeg_command = []

    return GoProPrepareReport(
        concat_list=str(concat_list_path),
        merged_output=str(merge_path),
        transcoded_output=str(final_path),
        files=[str(item.path) for item in files],
        ffmpeg_command=ffmpeg_command,
    )
