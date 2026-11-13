import re
import time

import httpx
from groq import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    Groq,
    RateLimitError,
)

from app.config import AppConfig
from app.providers.base import ProviderError, RequestTooLargeError
from app.providers.models import DEFAULT_ANALYSIS_MODEL


class GroqAnalyzer:
    def __init__(
        self, api_key: str, model: str = DEFAULT_ANALYSIS_MODEL, attempts: int = 3
    ) -> None:
        self._client = Groq(api_key=api_key, max_retries=0)
        self._model = model
        self._attempts = attempts

    def generate(self, prompt: str) -> str:
        for attempt in range(self._attempts):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Sigue las instrucciones del usuario y devuelve "
                                "únicamente JSON válido."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    reasoning_effort="low",
                    max_completion_tokens=2048,
                )
                if not response.choices:
                    raise ProviderError("Groq devolvió una respuesta vacía.")
                content = response.choices[0].message.content
                if not content:
                    raise ProviderError("Groq devolvió una respuesta vacía.")
                return content
            except AuthenticationError as exc:
                raise ProviderError(
                    "Groq rechazó la clave. Revísala en config.local.json."
                ) from exc
            except RateLimitError as exc:
                details = exc.message.casefold()
                if _requested_tokens_exceed_limit(details):
                    raise RequestTooLargeError(
                        "La ventana de transcripción supera el límite de tokens "
                        "por minuto de tu organización en Groq. AiClip la dividirá "
                        "en partes más pequeñas."
                    ) from exc
                limit_name = _rate_limit_name(details)
                if limit_name in {"requests per day", "tokens per day"}:
                    reset = _rate_limit_reset(
                        exc, "requests" if "requests" in limit_name else "tokens"
                    )
                    raise ProviderError(
                        f"Groq agotó el límite diario de {limit_name} para esta "
                        f"organización.{_reset_message(reset)} Revisa el uso y los "
                        "límites de la organización en console.groq.com/settings/limits."
                    ) from exc
                wait_seconds = _retry_delay(exc, attempt)
                if wait_seconds is None or wait_seconds > 60:
                    reset = _rate_limit_reset(
                        exc, "tokens" if "tokens" in details else "requests"
                    )
                    raise ProviderError(
                        f"Groq limitó las solicitudes ({limit_name or 'límite temporal'})."
                        f"{_reset_message(reset if reset is not None else wait_seconds)} "
                        "Espera ese tiempo y vuelve a intentarlo."
                    ) from exc
                if attempt + 1 == self._attempts:
                    raise ProviderError(
                        f"Groq mantuvo el límite de {limit_name or 'solicitudes'} "
                        f"después de esperar {_format_seconds(wait_seconds)}. "
                        "Espera a que se renueve e inténtalo de nuevo."
                    ) from exc
                time.sleep(wait_seconds)
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
                if exc.status_code == 400:
                    details = exc.message.casefold()
                    if any(
                        marker in details
                        for marker in (
                            "context length",
                            "context_length",
                            "too many tokens",
                            "maximum context",
                        )
                    ):
                        raise ProviderError(
                            "La transcripción es demasiado larga para analizarla "
                            "en una sola solicitud. Prueba con un vídeo más corto."
                        ) from exc
                    raise ProviderError(
                        "Groq rechazó la solicitud de análisis (HTTP 400). "
                        "Cierra y vuelve a abrir AiClip para usar la corrección "
                        "y vuelve a analizar el vídeo."
                    ) from exc
                if exc.status_code == 413:
                    raise RequestTooLargeError(
                        "La solicitud de análisis supera el tamaño permitido por Groq."
                    ) from exc
                raise ProviderError(
                    f"Groq no pudo analizar el texto (error {exc.status_code})."
                ) from exc
            except httpx.HTTPError as exc:
                if attempt + 1 == self._attempts:
                    raise ProviderError(
                        "No hay conexión con Groq. Comprueba Internet e inténtalo de nuevo."
                    ) from exc
                self._wait(attempt)

        raise ProviderError("Groq no pudo analizar el texto.")

    def close(self) -> None:
        self._client.close()

    @staticmethod
    def _wait(attempt: int) -> None:
        time.sleep(min(2**attempt, 8))


def _rate_limit_name(message: str) -> str:
    for name in (
        "requests per day",
        "tokens per day",
        "input tokens per minute",
        "output tokens per minute",
        "tokens per minute",
        "requests per minute",
        "(itpm)",
        "(otpm)",
        "(tpm)",
    ):
        if name in message:
            return {
                "(itpm)": "input tokens per minute",
                "(otpm)": "output tokens per minute",
                "(tpm)": "tokens per minute",
            }.get(name, name)
    return ""


def _requested_tokens_exceed_limit(message: str) -> bool:
    if not any(marker in message for marker in ("tokens per minute", "(tpm)", "itpm")):
        return False
    requested = re.search(r"requested\s+([\d,]+)", message)
    limit = re.search(r"(?:limit(?:\s+is)?|maximum)\s*:?\s*([\d,]+)", message)
    if not requested or not limit:
        return False
    return int(requested.group(1).replace(",", "")) > int(
        limit.group(1).replace(",", "")
    )


def _retry_delay(error: RateLimitError, attempt: int) -> float | None:
    retry_after = error.response.headers.get("retry-after")
    if retry_after:
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            parsed = _parse_reset_seconds(retry_after)
            if parsed is not None:
                return parsed
    reset_key = "tokens" if "tokens" in error.message.casefold() else "requests"
    reset = _rate_limit_reset(error, reset_key)
    if reset is not None:
        return reset
    retry_hint = re.search(r"try again in\s+([\d.]+)\s*(h|m|s)", error.message.casefold())
    if retry_hint:
        return float(retry_hint.group(1)) * {
            "h": 3600,
            "m": 60,
            "s": 1,
        }[retry_hint.group(2)]
    return float(min(2**attempt, 8))


def _rate_limit_reset(error: RateLimitError, limit: str) -> float | None:
    header = error.response.headers.get(f"x-ratelimit-reset-{limit}")
    return _parse_reset_seconds(header) if header else None


def _parse_reset_seconds(value: str) -> float | None:
    value = value.strip().casefold()
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    parts = re.findall(r"([\d.]+)\s*(h|m|s)", value)
    if not parts:
        return None
    multipliers = {"h": 3600, "m": 60, "s": 1}
    return sum(float(amount) * multipliers[unit] for amount, unit in parts)


def _format_seconds(seconds: float) -> str:
    total = max(1, round(seconds))
    if total < 60:
        return f"{total} segundo(s)"
    minutes, remainder = divmod(total, 60)
    if minutes < 60:
        return f"{minutes} min {remainder} s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes} min"


def _reset_message(seconds: float | None) -> str:
    return (
        f" Se renueva en aproximadamente {_format_seconds(seconds)}."
        if seconds is not None
        else ""
    )


def create_analyzer(config: AppConfig) -> GroqAnalyzer:
    if not config.groq_api_key:
        raise ProviderError("Falta groq_api_key en config.local.json.")
    if config.analysis_fallback:
        raise ProviderError(
            "El proveedor de fallback de análisis configurado no está disponible."
        )
    return GroqAnalyzer(config.groq_api_key, config.analysis_model)
