from dataclasses import dataclass
from typing import Protocol


class ProviderError(RuntimeError):
    """Error legible al usar un proveedor de IA."""


class RequestTooLargeError(ProviderError):
    """Error que permite reducir el tamaño de una solicitud al proveedor."""


@dataclass(frozen=True)
class TranscriptWord:
    start: float
    end: float
    text: str


class TranscriptionProvider(Protocol):
    def transcribe(
        self, audio_path: str, language: str | None = None
    ) -> list[TranscriptWord]: ...


class AnalysisProvider(Protocol):
    def generate(self, prompt: str) -> str: ...
