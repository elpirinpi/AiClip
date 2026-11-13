import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import httpx
from groq import APIStatusError, RateLimitError

import app.providers.groq_llm as groq_llm
from app.config import AppConfig
from app.providers.base import ProviderError, RequestTooLargeError
from app.providers.models import DEFAULT_ANALYSIS_MODEL


def _config(groq_api_key: str = "test-groq") -> AppConfig:
    return AppConfig(
        groq_api_key=groq_api_key,
        gemini_api_key="",
        ffmpeg_path="ffmpeg",
        output_folder=Path("output"),
    )


def test_groq_analyzer_uses_json_mode_and_returns_content(monkeypatch) -> None:
    request = {}

    class FakeClient:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self.create)
            )

        def create(self, **kwargs):
            request.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content='{"clips": []}')
                    )
                ]
            )

        def close(self):
            pass

    monkeypatch.setattr(groq_llm, "Groq", FakeClient)
    analyzer = groq_llm.create_analyzer(_config())

    assert analyzer.generate("Devuelve un objeto JSON.") == '{"clips": []}'
    analyzer.close()
    assert request["model"] == DEFAULT_ANALYSIS_MODEL
    assert request["response_format"] == {"type": "json_object"}
    assert request["reasoning_effort"] == "low"
    assert "temperature" not in request


def test_groq_analyzer_requires_only_the_groq_key() -> None:
    analyzer = groq_llm.create_analyzer(_config())

    assert isinstance(analyzer, groq_llm.GroqAnalyzer)
    analyzer.close()


def test_groq_analyzer_reports_missing_groq_key() -> None:
    with pytest.raises(ProviderError, match="Falta groq_api_key"):
        groq_llm.create_analyzer(_config(groq_api_key=""))


def _rate_limit_error(message: str, headers: dict[str, str] | None = None):
    response = httpx.Response(
        429,
        headers=headers,
        request=httpx.Request("POST", "https://api.groq.com"),
    )
    return RateLimitError(message, response=response, body={"error": {"message": message}})


def test_groq_tpm_request_over_limit_triggers_transcript_splitting(monkeypatch) -> None:
    error = _rate_limit_error(
        "Rate limit reached on tokens per minute (TPM): Limit 8000, "
        "Used 0, Requested 10000."
    )

    class FakeClient:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **_kwargs: (_ for _ in ()).throw(error)
                )
            )

        def close(self):
            pass

    monkeypatch.setattr(groq_llm, "Groq", FakeClient)
    analyzer = groq_llm.create_analyzer(_config())
    with pytest.raises(RequestTooLargeError, match="AiClip la dividirá"):
        analyzer.generate("transcripción")
    analyzer.close()


def test_groq_daily_quota_error_names_limit_and_reset(monkeypatch) -> None:
    error = _rate_limit_error(
        "Rate limit reached on requests per day (RPD).",
        {"x-ratelimit-reset-requests": "2h30m"},
    )

    class FakeClient:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **_kwargs: (_ for _ in ()).throw(error)
                )
            )

        def close(self):
            pass

    monkeypatch.setattr(groq_llm, "Groq", FakeClient)
    analyzer = groq_llm.create_analyzer(_config())
    with pytest.raises(ProviderError, match="límite diario de requests per day") as err:
        analyzer.generate("transcripción")
    assert "2 h 30 min" in str(err.value)
    analyzer.close()


def test_groq_rate_limit_waits_for_retry_after(monkeypatch) -> None:
    error = _rate_limit_error(
        "Rate limit reached on requests per minute (RPM).",
        {"retry-after": "5"},
    )
    waits = []

    class FakeClient:
        def __init__(self, **_kwargs):
            self.calls = 0

            def create(**_kwargs):
                self.calls += 1
                if self.calls < 2:
                    raise error
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content='{"clips": []}')
                        )
                    ]
                )

            self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))

        def close(self):
            pass

    monkeypatch.setattr(groq_llm, "Groq", FakeClient)
    monkeypatch.setattr(groq_llm.time, "sleep", waits.append)
    analyzer = groq_llm.create_analyzer(_config())

    assert analyzer.generate("transcripción") == '{"clips": []}'
    assert waits == [5]
    analyzer.close()


def test_groq_rate_limit_uses_retry_time_from_error_message(monkeypatch) -> None:
    error = _rate_limit_error(
        "Rate limit reached on tokens per minute (TPM). Please try again in 9.5s."
    )
    waits = []

    class FakeClient:
        def __init__(self, **_kwargs):
            self.calls = 0

            def create(**_kwargs):
                self.calls += 1
                if self.calls < 2:
                    raise error
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content='{"clips": []}')
                        )
                    ]
                )

            self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))

        def close(self):
            pass

    monkeypatch.setattr(groq_llm, "Groq", FakeClient)
    monkeypatch.setattr(groq_llm.time, "sleep", waits.append)
    analyzer = groq_llm.create_analyzer(_config())

    assert analyzer.generate("transcripción") == '{"clips": []}'
    assert waits == [9.5]
    analyzer.close()

@pytest.mark.parametrize(
    ("api_message", "expected_message"),
    [
        (
            "The input exceeds the maximum context length",
            "La transcripción es demasiado larga",
        ),
        (
            "Invalid request parameters",
            "Groq rechazó la solicitud de análisis (HTTP 400)",
        ),
        (
            "Request entity too large",
            "La solicitud de análisis supera el tamaño permitido por Groq",
        ),
    ],
    ids=["context-too-long", "bad-request", "payload-too-large"],
)
def test_groq_analyzer_explains_bad_request(
    monkeypatch, api_message: str, expected_message: str
) -> None:
    status_code = 413 if "too large" in api_message.casefold() else 400
    response = httpx.Response(
        status_code, request=httpx.Request("POST", "https://api.groq.com")
    )
    error = APIStatusError(api_message, response=response, body={"error": api_message})

    class FakeClient:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=lambda **_kwargs: (_ for _ in ()).throw(error))
            )

        def close(self):
            pass

    monkeypatch.setattr(groq_llm, "Groq", FakeClient)
    analyzer = groq_llm.create_analyzer(_config())
    with pytest.raises(ProviderError, match=re.escape(expected_message)):
        analyzer.generate("Devuelve un objeto JSON.")
    analyzer.close()
