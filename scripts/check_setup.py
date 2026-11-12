import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import httpx
from google import genai
from google.genai import errors as genai_errors
from groq import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    Groq,
    RateLimitError,
)

from app.config import AppConfig, ConfigError, load_config


def check_ffmpeg(config: AppConfig) -> bool:
    executable = shutil.which(config.ffmpeg_path)
    if executable is None and Path(config.ffmpeg_path).is_file():
        executable = config.ffmpeg_path
    if executable is None:
        print(
            "ERROR: FFmpeg no está disponible. Instálalo o indica su ruta "
            "en config.local.json."
        )
        return False

    try:
        result = subprocess.run(
            [executable, "-version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        print("ERROR: No se pudo ejecutar FFmpeg.")
        return False

    version = next(
        (line for line in result.stdout.splitlines() if line.startswith("ffmpeg")),
        "versión no identificada",
    )
    print(f"OK: FFmpeg disponible ({version}).")
    return True


def check_groq(api_key: str) -> bool:
    if not api_key:
        print("ERROR: Falta groq_api_key en config.local.json.")
        return False

    try:
        Groq(api_key=api_key).models.list()
    except AuthenticationError:
        print("ERROR: Groq rechazó la clave. Revísala en config.local.json.")
        return False
    except RateLimitError:
        print("ERROR: Groq alcanzó un límite de solicitudes o cuota.")
        return False
    except APIConnectionError:
        print("ERROR: No hay conexión con Groq. Comprueba Internet e inténtalo de nuevo.")
        return False
    except APIStatusError:
        print("ERROR: Groq no pudo validar la clave. Inténtalo más tarde.")
        return False

    print("OK: Groq aceptó la clave.")
    return True


def check_gemini(api_key: str) -> bool:
    if not api_key:
        print("ERROR: Falta gemini_api_key en config.local.json.")
        return False

    client = genai.Client(api_key=api_key)
    try:
        next(iter(client.models.list()), None)
    except genai_errors.APIError as exc:
        if exc.code in (401, 403):
            print("ERROR: Gemini rechazó la clave. Revísala en config.local.json.")
        elif exc.code == 429:
            print("ERROR: Gemini alcanzó un límite de solicitudes o cuota.")
        else:
            print("ERROR: Gemini no pudo validar la clave. Inténtalo más tarde.")
        return False
    except httpx.HTTPError:
        print("ERROR: No hay conexión con Gemini. Comprueba Internet e inténtalo de nuevo.")
        return False
    finally:
        client.close()

    print("OK: Gemini aceptó la clave.")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Comprueba FFmpeg y valida las claves de Groq y Gemini."
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Comprueba solo FFmpeg, sin conectar a los proveedores.",
    )
    args = parser.parse_args()

    try:
        config = load_config()
    except ConfigError as exc:
        print(f"ERROR: {exc}")
        return 1

    checks = [check_ffmpeg(config)]
    if args.local_only:
        print("AVISO: Las claves de API no se comprobaron.")
    else:
        checks.extend(
            [check_groq(config.groq_api_key), check_gemini(config.gemini_api_key)]
        )

    return 0 if all(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
