import json
import os
from pathlib import Path

import pytest

from app.config import ConfigError, load_config
from app.pipeline.segment import Clip
from app.pipeline.thumbnail import generate_thumbnail_text
from app.providers.base import TranscriptWord
from app.providers.groq_llm import create_analyzer
from app.providers.groq_stt import create_transcriber


pytestmark = pytest.mark.integration


def _integration_config():
    if os.environ.get("AICLIP_RUN_INTEGRATION") != "1":
        pytest.skip("Activa AICLIP_RUN_INTEGRATION=1 para ejecutar pruebas de API.")
    try:
        config = load_config()
    except ConfigError as exc:
        pytest.skip(f"No hay configuración local para las pruebas de API: {exc}")
    return config


def test_groq_returns_per_word_timestamps() -> None:
    config = _integration_config()
    if not config.groq_api_key:
        pytest.skip("Falta groq_api_key.")
    audio_path = os.environ.get("AICLIP_GROQ_TEST_AUDIO")
    if not audio_path or not Path(audio_path).is_file():
        pytest.skip("Define AICLIP_GROQ_TEST_AUDIO con un MP3 de voz de prueba.")

    words = create_transcriber(config).transcribe(audio_path)

    assert words
    assert all(word.text and word.start >= 0 and word.end >= word.start for word in words)


def test_groq_returns_a_json_clip_list() -> None:
    config = _integration_config()
    if not config.groq_api_key:
        pytest.skip("Falta groq_api_key.")
    prompt = (
        'Devuelve únicamente {"clips": []} como objeto JSON válido. '
        "No incluyas texto adicional."
    )

    analyzer = create_analyzer(config)
    try:
        response = analyzer.generate(prompt)
    finally:
        analyzer.close()

    payload = json.loads(response)
    assert isinstance(payload, dict)
    assert isinstance(payload.get("clips"), list)


def test_groq_generates_thumbnail_text_with_at_most_five_words() -> None:
    config = _integration_config()
    if not config.groq_api_key:
        pytest.skip("Falta groq_api_key.")

    analyzer = create_analyzer(load_config())
    try:
        text = generate_thumbnail_text(
            Clip(0, 40, "La pausa que cambia todo", "Escuchar mejor.", 9),
            [
                TranscriptWord(0, 2, "Una"),
                TranscriptWord(2, 4, "pausa"),
                TranscriptWord(4, 6, "puede"),
                TranscriptWord(6, 8, "cambiar"),
                TranscriptWord(8, 10, "una"),
                TranscriptWord(10, 12, "conversación."),
            ],
            analyzer,
        )
    finally:
        analyzer.close()

    assert 1 <= len(text.split()) <= 5
