import json
import math
from pathlib import Path
from typing import Any

from app.config import PROJECT_ROOT
from app.pipeline.segment import ClipCandidate
from app.providers.base import AnalysisProvider, ProviderError, TranscriptWord


HIGHLIGHTS_PROMPT_PATH = PROJECT_ROOT / "prompts" / "highlights.txt"


class AnalysisError(RuntimeError):
    """Error de análisis que se puede mostrar al usuario."""


def build_highlights_prompt(
    words: list[TranscriptWord], prompt_path: Path = HIGHLIGHTS_PROMPT_PATH
) -> str:
    try:
        template = prompt_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise AnalysisError("No se encontró el prompt para seleccionar momentos.") from exc
    transcript = "\n".join(
        f"[{word.start:.2f}-{word.end:.2f}] {word.text}" for word in words
    )
    return f"{template.strip()}\n\nTRANSCRIPCIÓN:\n{transcript}"


def analyze_transcript(
    words: list[TranscriptWord], provider: AnalysisProvider
) -> list[ClipCandidate]:
    if not words:
        return []
    prompt = build_highlights_prompt(words)
    try:
        response = provider.generate(prompt)
    except ProviderError:
        raise
    except Exception as exc:
        raise AnalysisError("No se pudieron analizar los momentos del vídeo.") from exc

    try:
        payload: Any = json.loads(response)
    except json.JSONDecodeError as exc:
        raise AnalysisError(
            "Gemini no devolvió JSON válido para los momentos del vídeo."
        ) from exc

    if not isinstance(payload, list):
        raise AnalysisError("La respuesta de Gemini debe ser una lista JSON de clips.")

    candidates: list[ClipCandidate] = []
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise AnalysisError(f"El clip {index} de Gemini no tiene un formato válido.")
        try:
            start = _number(item["start"], "start")
            end = _number(item["end"], "end")
            title = _string(item["title"], "title")
            reason = _string(item["reason"], "reason")
            score_value = _number(item["score"], "score")
        except (KeyError, ValueError, TypeError) as exc:
            raise AnalysisError(f"El clip {index} de Gemini tiene campos inválidos.") from exc
        if not math.isfinite(start) or not math.isfinite(end) or end <= start or start < 0:
            raise AnalysisError(f"El clip {index} tiene marcas de tiempo inválidas.")
        if not math.isfinite(score_value) or not 1 <= score_value <= 10 or not score_value.is_integer():
            raise AnalysisError(f"La puntuación del clip {index} debe ser un entero de 1 a 10.")
        title = " ".join(title.split()[:8])
        if not title or start < 0 or end <= start:
            raise AnalysisError(f"El clip {index} tiene un intervalo o título inválido.")
        candidates.append(
            ClipCandidate(
                start=start,
                end=end,
                title=title,
                reason=reason,
                score=int(score_value),
            )
        )
    return candidates


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} debe ser un número.")
    return float(value)


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} debe ser texto.")
    return value.strip()
