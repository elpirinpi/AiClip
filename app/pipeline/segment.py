import math
from dataclasses import dataclass

from app.providers.base import TranscriptWord


MIN_CLIP_SECONDS = 30.0
MAX_CLIP_SECONDS = 90.0


@dataclass(frozen=True)
class ClipCandidate:
    start: float
    end: float
    title: str
    reason: str
    score: int


@dataclass(frozen=True)
class Clip(ClipCandidate):
    pass


def snap_clips_to_sentences(
    candidates: list[ClipCandidate],
    words: list[TranscriptWord],
    min_seconds: float = MIN_CLIP_SECONDS,
    max_seconds: float = MAX_CLIP_SECONDS,
) -> tuple[list[Clip], list[str]]:
    if not words:
        return [], ["No hay palabras transcritas para ajustar los cortes."]
    ordered_words = sorted(words, key=lambda word: (word.start, word.end))
    sentence_starts = [ordered_words[0].start]
    sentence_ends: list[float] = []
    for index, word in enumerate(ordered_words):
        if word.text.rstrip().endswith((".", "?", "!")):
            sentence_ends.append(word.end)
            if index + 1 < len(ordered_words):
                sentence_starts.append(ordered_words[index + 1].start)
    if not sentence_ends or sentence_ends[-1] < ordered_words[-1].end:
        sentence_ends.append(ordered_words[-1].end)

    selected: list[Clip] = []
    warnings: list[str] = []
    for candidate in sorted(candidates, key=lambda clip: clip.score, reverse=True):
        start = max(
            (boundary for boundary in sentence_starts if boundary <= candidate.start),
            default=sentence_starts[0],
        )
        end = min(
            (boundary for boundary in sentence_ends if boundary >= candidate.end),
            default=sentence_ends[-1],
        )
        duration = end - start
        if duration < min_seconds or duration > max_seconds:
            warnings.append(
                f"Se descartó «{candidate.title}»: al ajustar las frases duraría "
                f"{duration:.1f} segundos."
            )
            continue
        if any(start < clip.end and end > clip.start for clip in selected):
            warnings.append(f"Se descartó «{candidate.title}» porque se solapa con otro clip.")
            continue
        selected.append(
            Clip(
                start=start,
                end=end,
                title=candidate.title,
                reason=candidate.reason,
                score=candidate.score,
            )
        )
    selected.sort(key=lambda clip: clip.score, reverse=True)
    return selected, warnings


def validate_clip_candidate(candidate: ClipCandidate) -> bool:
    return (
        math.isfinite(candidate.start)
        and math.isfinite(candidate.end)
        and candidate.start >= 0
        and candidate.end > candidate.start
        and 1 <= candidate.score <= 10
    )
