from pathlib import Path

import pytest

import app.pipeline.audio as audio
from app.pipeline.audio import MAX_AUDIO_CHUNK_BYTES, plan_audio_chunks


def test_audio_chunks_are_under_30_minutes_and_overlap() -> None:
    chunks = plan_audio_chunks(3601, chunk_seconds=1800, overlap_seconds=2)

    assert chunks == [(0.0, 1800), (1798, 3600), (3598, 3601)]
    assert chunks[1][0] < chunks[0][1]
    assert chunks[2][0] < chunks[1][1]
    assert MAX_AUDIO_CHUNK_BYTES < 25_000_000


def test_find_ffprobe_uses_path_when_not_next_to_ffmpeg(
    monkeypatch, tmp_path: Path
) -> None:
    ffmpeg = tmp_path / "ffmpeg.exe"
    ffprobe = tmp_path / "ffprobe.exe"
    ffmpeg.touch()
    ffprobe.touch()
    monkeypatch.setattr(audio.shutil, "which", lambda _name: str(ffprobe))

    assert audio._find_ffprobe(str(ffmpeg)) == str(ffprobe)


@pytest.mark.parametrize(
    ("duration", "chunk_seconds", "overlap"),
    [(0, 1800, 2), (60, 0, 2), (60, 10, 10), (60, 10, -1)],
)
def test_audio_chunk_plan_rejects_invalid_values(
    duration: float, chunk_seconds: float, overlap: float
) -> None:
    with pytest.raises(ValueError):
        plan_audio_chunks(duration, chunk_seconds, overlap)
