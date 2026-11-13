import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.providers.models import DEFAULT_ANALYSIS_MODEL


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DATA_DIR = (
    Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    / "AiClip"
    if getattr(sys, "frozen", False)
    else PROJECT_ROOT
)
DEFAULT_CONFIG_PATH = APP_DATA_DIR / "config.local.json"
DEFAULT_OUTPUT_FOLDER = APP_DATA_DIR / "output"


class ConfigError(ValueError):
    """Error de configuración que se puede mostrar directamente al usuario."""


@dataclass(frozen=True)
class AppConfig:
    groq_api_key: str = field(repr=False)
    gemini_api_key: str = field(repr=False)
    ffmpeg_path: str
    output_folder: Path
    transcription_model: str = "whisper-large-v3-turbo"
    analysis_model: str = DEFAULT_ANALYSIS_MODEL
    subtitle_style: str = "classic"
    reframe_mode: str = "center"
    transcription_fallback: str | None = None
    analysis_fallback: str | None = None


def load_config(path: Path | None = None) -> AppConfig:
    config_path = path or DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise ConfigError(
            f"No se encontró {config_path}. Copia config.example.json a "
            "config.local.json y añade tus claves."
        )

    try:
        data: Any = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"El archivo {config_path} no contiene JSON válido "
            f"(línea {exc.lineno}, columna {exc.colno})."
        ) from exc
    except OSError as exc:
        raise ConfigError(f"No se pudo leer el archivo {config_path}.") from exc

    if not isinstance(data, dict):
        raise ConfigError("La configuración debe ser un objeto JSON.")
    return load_config_data(data)


def save_settings(
    *,
    groq_api_key: str | None = None,
    gemini_api_key: str | None = None,
    output_folder: str | None = None,
    subtitle_style: str | None = None,
    reframe_mode: str | None = None,
    path: Path | None = None,
) -> AppConfig:
    config_path = path or DEFAULT_CONFIG_PATH
    try:
        data: Any = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        data = {}
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError("No se pudo leer la configuración local.") from exc
    if not isinstance(data, dict):
        raise ConfigError("La configuración debe ser un objeto JSON.")

    if groq_api_key is not None and groq_api_key.strip():
        data["groq_api_key"] = groq_api_key.strip()
    if gemini_api_key is not None and gemini_api_key.strip():
        data["gemini_api_key"] = gemini_api_key.strip()
    if output_folder is not None:
        if not output_folder.strip():
            raise ConfigError("La carpeta de salida no puede estar vacía.")
        data["output_folder"] = output_folder.strip()
    if subtitle_style is not None:
        data["subtitle_style"] = subtitle_style
    if reframe_mode is not None:
        data["reframe_mode"] = reframe_mode

    config = load_config_data(data)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = config_path.with_suffix(config_path.suffix + ".tmp")
    try:
        temporary_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(config_path)
    except OSError as exc:
        temporary_path.unlink(missing_ok=True)
        raise ConfigError("No se pudo guardar la configuración local.") from exc
    return config


def load_config_data(data: dict[str, Any]) -> AppConfig:
    groq_api_key = _string_setting(data, "groq_api_key", "")
    gemini_api_key = _string_setting(data, "gemini_api_key", "")
    ffmpeg_path = _string_setting(data, "ffmpeg_path", "ffmpeg")
    output_folder_value = _string_setting(
        data, "output_folder", str(DEFAULT_OUTPUT_FOLDER)
    )
    transcription_model = _string_setting(
        data, "transcription_model", "whisper-large-v3-turbo"
    )
    analysis_model = _string_setting(data, "analysis_model", DEFAULT_ANALYSIS_MODEL)
    if analysis_model.startswith("gemini-") or analysis_model == "openai/gpt-oss-120b":
        analysis_model = DEFAULT_ANALYSIS_MODEL
    subtitle_style = _string_setting(data, "subtitle_style", "classic")
    reframe_mode = _string_setting(data, "reframe_mode", "center")
    if subtitle_style not in {"classic", "impact"}:
        raise ConfigError("subtitle_style debe ser classic o impact.")
    if reframe_mode not in {"center", "face", "blur"}:
        raise ConfigError("reframe_mode debe ser center, face o blur.")
    return AppConfig(
        groq_api_key=groq_api_key.strip(),
        gemini_api_key=gemini_api_key.strip(),
        ffmpeg_path=ffmpeg_path.strip() or "ffmpeg",
        output_folder=_resolve_path(output_folder_value),
        transcription_model=transcription_model.strip(),
        analysis_model=analysis_model.strip(),
        subtitle_style=subtitle_style,
        reframe_mode=reframe_mode,
        transcription_fallback=_optional_string_setting(
            data, "transcription_fallback"
        ),
        analysis_fallback=_optional_string_setting(data, "analysis_fallback"),
    )


def _string_setting(data: dict[str, Any], name: str, default: str) -> str:
    value = data.get(name, default)
    if not isinstance(value, str):
        raise ConfigError(f"El valor '{name}' debe ser texto.")
    return value


def _optional_string_setting(data: dict[str, Any], name: str) -> str | None:
    value = data.get(name)
    if value is not None and not isinstance(value, str):
        raise ConfigError(f"El valor '{name}' debe ser texto o null.")
    return value.strip() if isinstance(value, str) and value.strip() else None


def _resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else APP_DATA_DIR / path
