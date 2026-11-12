from pathlib import Path

from app.pipeline.audio import AudioChunk
from app.pipeline.transcribe import transcribe_chunks
from app.providers.base import TranscriptWord


class FakeProvider:
    def __init__(self) -> None:
        self.responses = {
            "first.mp3": [
                TranscriptWord(4.1, 4.3, "The"),
                TranscriptWord(4.4, 4.7, "idea"),
                TranscriptWord(5.0, 5.3, "works."),
            ],
            "second.mp3": [
                TranscriptWord(0.1, 0.3, "The"),
                TranscriptWord(0.4, 0.7, "idea"),
                TranscriptWord(1.0, 1.3, "greatly"),
                TranscriptWord(1.4, 1.7, "improved."),
            ],
        }

    def transcribe(
        self, audio_path: str, language: str | None = None
    ) -> list[TranscriptWord]:
        return self.responses[Path(audio_path).name]


def test_transcribe_chunks_offsets_timestamps_and_removes_overlap_duplicates() -> None:
    chunks = [
        AudioChunk(Path("first.mp3"), start=0, end=6),
        AudioChunk(Path("second.mp3"), start=4, end=10),
    ]

    words = transcribe_chunks(chunks, FakeProvider())

    assert [word.text for word in words] == [
        "The",
        "idea",
        "works.",
        "greatly",
        "improved.",
    ]
    assert words[-1].start == 5.4
    assert words[-1].end == 5.7


def test_transcribe_chunks_preserves_empty_results() -> None:
    class EmptyProvider:
        def transcribe(
            self, audio_path: str, language: str | None = None
        ) -> list[TranscriptWord]:
            return []

    assert transcribe_chunks([], EmptyProvider()) == []
