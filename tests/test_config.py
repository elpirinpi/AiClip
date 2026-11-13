import json
from pathlib import Path

import pytest

from app.config import AppConfig, ConfigError, load_config
from app.providers.models import DEFAULT_ANALYSIS_MODEL


def test_load_config_uses_project_relative_output_path(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "groq_api_key": "test-groq",
                "gemini_api_key": "test-gemini",
                "ffmpeg_path": "ffmpeg",
                "output_folder": "clips",
                "transcription_fallback": None,
                "analysis_fallback": "another-provider",
            }
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert isinstance(config, AppConfig)
    assert config.groq_api_key == "test-groq"
    assert config.gemini_api_key == "test-gemini"
    assert config.output_folder.name == "clips"
    assert config.transcription_fallback is None
    assert config.analysis_fallback == "another-provider"


def test_load_config_rejects_non_string_api_key(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text('{"groq_api_key": 123}', encoding="utf-8")

    with pytest.raises(ConfigError, match="groq_api_key"):
        load_config(config_path)


def test_load_config_replaces_legacy_gemini_model_with_groq_model(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "groq_api_key": "test-groq",
                "ffmpeg_path": "ffmpeg",
                "output_folder": "clips",
                "analysis_model": "gemini-2.5-flash",
            }
        ),
        encoding="utf-8",
    )

    assert load_config(config_path).analysis_model == DEFAULT_ANALYSIS_MODEL


def test_load_config_migrates_slow_groq_analysis_model(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "groq_api_key": "test-groq",
                "ffmpeg_path": "ffmpeg",
                "output_folder": "clips",
                "analysis_model": "openai/gpt-oss-120b",
            }
        ),
        encoding="utf-8",
    )

    assert load_config(config_path).analysis_model == DEFAULT_ANALYSIS_MODEL


def test_app_config_repr_does_not_expose_api_keys() -> None:
    config = AppConfig(
        groq_api_key="secret-groq",
        gemini_api_key="secret-gemini",
        ffmpeg_path="ffmpeg",
        output_folder=Path("output"),
    )

    assert "secret-groq" not in repr(config)
    assert "secret-gemini" not in repr(config)
