import re
from collections.abc import Sequence

from app.pipeline.audio import AudioChunk
from app.providers.base import TranscriptWord, TranscriptionProvider


def transcribe_chunks(
    chunks: Sequence[AudioChunk],
    provider: TranscriptionProvider,
    language: str | None = None,
) -> list[TranscriptWord]:
    merged: list[TranscriptWord] = []
    for chunk in chunks:
        chunk_words = provider.transcribe(str(chunk.path), language=language)
        adjusted = [
            TranscriptWord(
                start=word.start + chunk.start,
                end=word.end + chunk.start,
                text=word.text,
            )
            for word in chunk_words
            if word.end >= word.start and word.text.strip()
        ]
        merged = _merge_overlap(merged, adjusted, chunk.start)
    return merged


def _merge_overlap(
    previous: list[TranscriptWord],
    current: list[TranscriptWord],
    overlap_start: float,
) -> list[TranscriptWord]:
    if not previous or not current:
        return previous + current

    previous_end = previous[-1].end
    previous_overlap = [
        word for word in previous if word.end >= overlap_start and word.start < previous_end
    ]
    current_overlap = [
        word for word in current if word.start < previous_end and word.end >= overlap_start
    ]
    duplicate_count = 0
    for left_start in range(len(previous_overlap)):
        max_count = min(
            len(previous_overlap) - left_start,
            len(current_overlap),
        )
        for count in range(max_count, duplicate_count, -1):
            left = previous_overlap[left_start : left_start + count]
            right = current_overlap[:count]
            if all(
                _normalize_word(a.text) == _normalize_word(b.text)
                and abs(a.start - b.start) <= 1.0
                for a, b in zip(left, right)
            ):
                duplicate_count = count
                break
    if duplicate_count:
        duplicates = set(id(word) for word in current_overlap[:duplicate_count])
        current = [word for word in current if id(word) not in duplicates]
    return previous + current


def _normalize_word(word: str) -> str:
    return re.sub(r"[^\w']+", "", word.casefold())
