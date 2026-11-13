from pathlib import Path
import sys

import app.pipeline.ffmpeg as ffmpeg


def test_find_ffmpeg_uses_configured_executable(monkeypatch, tmp_path: Path) -> None:
    executable = tmp_path / "custom-ffmpeg.exe"
    executable.touch()
    monkeypatch.setattr(ffmpeg.shutil, "which", lambda _value: None)

    assert ffmpeg.find_ffmpeg(str(executable)) == str(executable)


def test_find_ffmpeg_falls_back_to_bundled_executable(
    monkeypatch, tmp_path: Path
) -> None:
    executable = tmp_path / "ffmpeg.exe"
    executable.touch()
    monkeypatch.setattr(ffmpeg.shutil, "which", lambda _value: None)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)

    assert ffmpeg.find_ffmpeg("ffmpeg") == str(executable)


def test_find_ffmpeg_reports_missing_when_not_installed(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(ffmpeg.shutil, "which", lambda _value: None)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)

    assert ffmpeg.find_ffmpeg("ffmpeg") is None
