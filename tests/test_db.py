import sqlite3
from pathlib import Path

from app.db import get_transcript_words, save_clips, save_results
from app.pipeline.segment import Clip
from app.providers.base import TranscriptWord


def test_save_clips_creates_schema_and_persists_results(tmp_path: Path) -> None:
    database = tmp_path / "nested" / "clips.sqlite3"
    clip = Clip(0, 35, "A moment", "A reason.", 8)

    saved = save_clips("local-video.mp4", [clip], database)

    assert saved == 1
    with sqlite3.connect(database) as connection:
        row = connection.execute(
            "SELECT source, start, end, title, reason, score FROM clips"
        ).fetchone()
    assert row == ("local-video.mp4", 0, 35, "A moment", "A reason.", 8)


def test_save_results_persists_transcript_for_rendering(tmp_path: Path) -> None:
    database = tmp_path / "results.sqlite3"
    clip = Clip(0, 35, "A moment", "A reason.", 8)
    words = [
        TranscriptWord(start=0.1, end=0.3, text="Hello"),
        TranscriptWord(start=0.4, end=0.8, text="world."),
    ]

    assert save_results("video.mp4", [clip], words, database) == 1
    assert get_transcript_words("video.mp4", database) == words

    assert save_results("video.mp4", [], [], database) == 0
    assert get_transcript_words("video.mp4", database) == []
