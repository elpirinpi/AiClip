import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from groq import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    Groq,
    RateLimitError,
)

from app.config import AppConfig
from app.providers.base import ProviderError, TranscriptWord


class GroqTranscriber:
    def __init__(
        self, api_key: str, model: str = "whisper-large-v3-turbo", attempts: int = 3
    ) -> None:
        self._client = Groq(api_key=api_key, max_retries=0)
        self._model = model
        self._attempts = attempts

    def transcribe(
        self, audio_path: str, language: str | None = None
    ) -> list[TranscriptWord]:
        path = Path(audio_path)
        if not path.is_file():
            raise ProviderError("No se encontró el fragmento de audio para transcribir.")

        for attempt in range(self._attempts):
            try:
                with path.open("rb") as audio_file:
                    response = self._client.audio.transcriptions.create(
                        file=audio_file,
                        model=self._model,
                        response_format="verbose_json",
                        timestamp_granularities=["word"],
                        language=language,
                    )
                words = getattr(response, "words", None)
                if not words:
                    raise ProviderError(
                        "Groq no devolvió marcas de tiempo por palabra. "
                        "No se puede continuar con los subtítulos precisos."
                    )
                parsed_words = []
                for word in words:
                    text = _word_field(word, "word")
                    start = _word_field(word, "start")
                    end = _word_field(word, "end")
                    try:
                        start_seconds = float(start)
                        end_seconds = float(end)
                    except (TypeError, ValueError) as exc:
                        raise ProviderError(
                            "Groq devolvió una palabra sin marcas de tiempo válidas."
                        ) from exc
                    if text and str(text).strip() and end_seconds >= start_seconds:
                        parsed_words.append(
                            TranscriptWord(
                                start=start_seconds,
                                end=end_seconds,
                                text=str(text).strip(),
                            )
                        )
                if not parsed_words:
                    raise ProviderError(
                        "Groq no devolvió marcas de tiempo por palabra válidas."
                    )
                return parsed_words
            except AuthenticationError as exc:
                raise ProviderError(
                    "Groq rechazó la clave. Revísala en config.local.json."
                ) from exc
            except RateLimitError as exc:
                if attempt + 1 == self._attempts:
                    raise ProviderError(
                        "Groq alcanzó su límite o cuota. Espera y vuelve a intentarlo."
                    ) from exc
                self._wait(attempt)
            except APIConnectionError as exc:
                if attempt + 1 == self._attempts:
                    raise ProviderError(
                        "No hay conexión con Groq. Comprueba Internet e inténtalo de nuevo."
                    ) from exc
                self._wait(attempt)
            except APIStatusError as exc:
                if exc.status_code >= 500 and attempt + 1 < self._attempts:
                    self._wait(attempt)
                    continue
                raise ProviderError(
                    f"Groq no pudo transcribir el audio (error {exc.status_code})."
                ) from exc

        raise ProviderError("Groq no pudo transcribir el audio.")

    @staticmethod
    def _wait(attempt: int) -> None:
        time.sleep(min(2**attempt, 8))


def _word_field(word: Any, name: str) -> Any:
    if isinstance(word, Mapping):
        return word.get(name)
    return getattr(word, name, None)


def create_transcriber(config: AppConfig) -> GroqTranscriber:
    if not config.groq_api_key:
        raise ProviderError("Falta groq_api_key en config.local.json.")
    if config.transcription_fallback:
        raise ProviderError(
            "El proveedor de fallback de transcripción configurado no está disponible."
        )
    return GroqTranscriber(config.groq_api_key, config.transcription_model)
