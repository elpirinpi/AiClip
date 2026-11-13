import json
import math
import re
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from app.config import PROJECT_ROOT
from app.pipeline.segment import ClipCandidate
from app.providers.base import (
    AnalysisProvider,
    ProviderError,
    RequestTooLargeError,
    TranscriptWord,
)


HIGHLIGHTS_PROMPT_PATH = PROJECT_ROOT / "prompts" / "highlights.txt"
MIN_CANDIDATE_SECONDS = 30.0
MAX_CANDIDATE_SECONDS = 90.0
TARGET_SECONDS_PER_CLIP = 180.0
MAX_TARGET_CLIPS = 10
MAX_CANDIDATES = 20
HOOK_WORDS = {
    "atención", "cuidado", "descubre", "error", "jamás", "mira", "nunca",
    "problema", "secreto", "sorprendente", "truco", "verdad",
}


class AnalysisError(RuntimeError):
    """Error de análisis que se puede mostrar al usuario."""


@dataclass(frozen=True)
class MomentCandidate:
    candidate_id: int
    start: float
    end: float
    text: str
    local_score: float


def build_highlights_prompt(
    candidates: list[MomentCandidate],
    target_clips: int,
    prompt_path: Path = HIGHLIGHTS_PROMPT_PATH,
) -> str:
    try:
        template = prompt_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise AnalysisError("No se encontró el prompt para seleccionar momentos.") from exc
    sections = [
        (
            f"[CANDIDATO {candidate.candidate_id} | "
            f"{candidate.start:.2f}-{candidate.end:.2f} s | "
            f"puntuación preliminar {candidate.local_score:.1f}/10]\n"
            f"{candidate.text}"
        )
        for candidate in candidates
    ]
    clip_word = "clip" if target_clips == 1 else "clips"
    return (
        f"{template.strip()}\n\nIntenta seleccionar hasta {target_clips} "
        f"{clip_word}.\n\n"
        + "\n\n".join(sections)
    )


def analyze_transcript(
    words: list[TranscriptWord],
    provider: AnalysisProvider,
    progress_callback: Callable[[int], None] | None = None,
) -> list[ClipCandidate]:
    if not words:
        return []
    ordered_words = sorted(words, key=lambda word: (word.start, word.end))
    duration = max(word.end for word in ordered_words) - ordered_words[0].start
    target_clips = _target_clip_count(duration)
    candidates = _generate_candidates(ordered_words, target_clips)
    if not candidates:
        return []
    if progress_callback:
        progress_callback(76)
    if progress_callback:
        progress_callback(80)
    selected = _rank_candidates(candidates, target_clips, provider)
    if progress_callback:
        progress_callback(90)
    return _deduplicate_candidates(selected)[:target_clips]


def _target_clip_count(duration: float) -> int:
    return min(MAX_TARGET_CLIPS, max(1, math.ceil(duration / TARGET_SECONDS_PER_CLIP)))


def _generate_candidates(
    words: list[TranscriptWord], target_clips: int
) -> list[MomentCandidate]:
    sentences = _sentences(words)
    if not sentences:
        return []

    sentence_ends = [sentence[-1].end for sentence in sentences]
    proposed: dict[tuple[float, float], MomentCandidate] = {}
    for start_index, sentence in enumerate(sentences):
        start = sentence[0].start
        for desired_duration in (45.0, 60.0, 75.0):
            end_index = bisect_left(sentence_ends, start + desired_duration, lo=start_index)
            if end_index >= len(sentences):
                continue
            end = sentence_ends[end_index]
            duration = end - start
            if not MIN_CANDIDATE_SECONDS <= duration <= MAX_CANDIDATE_SECONDS:
                continue
            candidate_words = [
                word for part in sentences[start_index : end_index + 1] for word in part
            ]
            text = " ".join(word.text.strip() for word in candidate_words).strip()
            score = _score_candidate(text, len(candidate_words), duration)
            key = (round(start, 2), round(end, 2))
            proposed[key] = MomentCandidate(0, start, end, text, score)

    target_limit = min(MAX_CANDIDATES, max(6, target_clips * 2))
    ranked = sorted(
        proposed.values(), key=lambda item: (-item.local_score, item.start, item.end)
    )
    selected: list[MomentCandidate] = []
    duration = max(sentence[-1].end for sentence in sentences) - sentences[0][0].start
    bin_width = duration / target_limit
    bins: list[list[MomentCandidate]] = [[] for _ in range(target_limit)]
    for candidate in ranked:
        center = (candidate.start + candidate.end) / 2
        bin_index = min(target_limit - 1, int((center - sentences[0][0].start) / bin_width))
        bins[bin_index].append(candidate)
    for candidates_in_bin in bins:
        for candidate in candidates_in_bin:
            if not any(_overlap_ratio(candidate, existing) > 0 for existing in selected):
                selected.append(candidate)
                break
        if len(selected) >= target_limit:
            break
    for candidate in ranked:
        if len(selected) >= target_limit:
            break
        if any(_overlap_ratio(candidate, existing) > 0 for existing in selected):
            continue
        selected.append(candidate)
    selected.sort(key=lambda item: item.start)
    return [
        MomentCandidate(index, item.start, item.end, item.text, item.local_score)
        for index, item in enumerate(selected, start=1)
    ]


def _sentences(words: list[TranscriptWord]) -> list[list[TranscriptWord]]:
    sentences: list[list[TranscriptWord]] = []
    current: list[TranscriptWord] = []
    for word in words:
        if not word.text.strip() or word.end < word.start:
            continue
        if current and (
            word.start - current[-1].end > 0.9
            or current[-1].text.rstrip().endswith((".", "?", "!"))
        ):
            sentences.append(current)
            current = []
        current.append(word)
    if current:
        sentences.append(current)
    return sentences


def _score_candidate(text: str, word_count: int, duration: float) -> float:
    tokens = re.findall(r"[a-záéíóúñü]+", text.casefold())
    density = min(1.0, word_count / max(1.0, duration * 2.0))
    hooks = min(1.0, sum(token in HOOK_WORDS for token in tokens) / 3.0)
    emphasis = min(1.0, (text.count("?") + text.count("!")) / 2.0)
    return 10 * (0.5 * density + 0.3 * hooks + 0.2 * emphasis)


def _overlap_ratio(first: MomentCandidate, second: MomentCandidate) -> float:
    overlap = max(0.0, min(first.end, second.end) - max(first.start, second.start))
    shorter = min(first.end - first.start, second.end - second.start)
    return overlap / shorter if shorter > 0 else 0.0


def _rank_candidates(
    candidates: list[MomentCandidate],
    target_clips: int,
    provider: AnalysisProvider,
) -> list[ClipCandidate]:
    try:
        response = provider.generate(
            build_highlights_prompt(candidates, target_clips)
        )
    except RequestTooLargeError as exc:
        if len(candidates) < 2:
            raise RequestTooLargeError(
                "Groq sigue rechazando un candidato individual por su tamaño. "
                "No se pudo completar el análisis."
            ) from exc
        middle = len(candidates) // 2
        return _rank_candidates(candidates[:middle], target_clips, provider) + (
            _rank_candidates(candidates[middle:], target_clips, provider)
        )
    except ProviderError:
        raise
    except Exception as exc:
        raise AnalysisError("No se pudieron analizar los momentos del vídeo.") from exc
    return _parse_ranked_candidates(response, candidates)


def _parse_ranked_candidates(
    response: str, candidates: list[MomentCandidate]
) -> list[ClipCandidate]:
    try:
        payload: Any = json.loads(response)
    except json.JSONDecodeError as exc:
        raise AnalysisError(
            "Groq no devolvió JSON válido para los momentos del vídeo."
        ) from exc
    if isinstance(payload, dict):
        payload = payload.get("clips")
    if not isinstance(payload, list):
        raise AnalysisError(
            "La respuesta de Groq debe contener una lista JSON de clips."
        )

    candidates_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    selected: list[ClipCandidate] = []
    used_ids: set[int] = set()
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise AnalysisError(f"El clip {index} de Groq no tiene un formato válido.")
        try:
            candidate_id = _integer(item["candidate_id"], "candidate_id")
            title = _string(item["title"], "title")
            reason = _string(item["reason"], "reason")
            score = _integer(item["score"], "score")
        except (KeyError, ValueError, TypeError) as exc:
            raise AnalysisError(f"El clip {index} de Groq tiene campos inválidos.") from exc
        if candidate_id not in candidates_by_id:
            raise AnalysisError(f"Groq seleccionó un candidato desconocido ({candidate_id}).")
        if candidate_id in used_ids:
            continue
        if not 1 <= score <= 10:
            raise AnalysisError(f"La puntuación del clip {index} debe estar entre 1 y 10.")
        candidate = candidates_by_id[candidate_id]
        used_ids.add(candidate_id)
        selected.append(
            ClipCandidate(
                start=candidate.start,
                end=candidate.end,
                title=" ".join(title.split()[:8]),
                reason=reason,
                score=score,
            )
        )
    return selected


def _deduplicate_candidates(
    candidates: list[ClipCandidate],
) -> list[ClipCandidate]:
    unique: list[ClipCandidate] = []
    for candidate in sorted(
        candidates, key=lambda clip: (-clip.score, clip.start, clip.end)
    ):
        if any(_same_moment(existing, candidate) for existing in unique):
            continue
        unique.append(candidate)
    return unique


def _same_moment(first: ClipCandidate, second: ClipCandidate) -> bool:
    overlap = max(0.0, min(first.end, second.end) - max(first.start, second.start))
    shorter_duration = min(first.end - first.start, second.end - second.start)
    return (
        shorter_duration > 0
        and overlap / shorter_duration >= 0.8
        and abs(first.start - second.start) <= 20
    )


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} debe ser un entero.")
    return value


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} debe ser texto.")
    return value.strip()
