import time

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from app.config import AppConfig
from app.providers.base import ProviderError


class GeminiAnalyzer:
    def __init__(
        self, api_key: str, model: str = "gemini-2.5-flash", attempts: int = 3
    ) -> None:
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._attempts = attempts

    def generate(self, prompt: str) -> str:
        for attempt in range(self._attempts):
            try:
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0,
                    ),
                )
                if not response.text:
                    raise ProviderError("Gemini devolvió una respuesta vacía.")
                return response.text
            except genai_errors.APIError as exc:
                if exc.code in (401, 403):
                    raise ProviderError(
                        "Gemini rechazó la clave. Revísala en config.local.json."
                    ) from exc
                if exc.code == 429 and attempt + 1 < self._attempts:
                    self._wait(attempt)
                    continue
                if exc.code == 429:
                    raise ProviderError(
                        "Gemini alcanzó su límite o cuota. Espera y vuelve a intentarlo."
                    ) from exc
                if exc.code >= 500 and attempt + 1 < self._attempts:
                    self._wait(attempt)
                    continue
                raise ProviderError(
                    f"Gemini no pudo analizar el texto (error {exc.code})."
                ) from exc
            except httpx.HTTPError as exc:
                if attempt + 1 == self._attempts:
                    raise ProviderError(
                        "No hay conexión con Gemini. Comprueba Internet e inténtalo de nuevo."
                    ) from exc
                self._wait(attempt)

        raise ProviderError("Gemini no pudo analizar el texto.")

    def close(self) -> None:
        self._client.close()

    @staticmethod
    def _wait(attempt: int) -> None:
        time.sleep(min(2**attempt, 8))


def create_analyzer(config: AppConfig) -> GeminiAnalyzer:
    if not config.gemini_api_key:
        raise ProviderError("Falta gemini_api_key en config.local.json.")
    if config.analysis_fallback:
        raise ProviderError(
            "El proveedor de fallback de análisis configurado no está disponible."
        )
    return GeminiAnalyzer(config.gemini_api_key, config.analysis_model)
