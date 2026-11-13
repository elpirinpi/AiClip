import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from app.config import AppConfig
from app.pipeline.ffmpeg import find_ffmpeg


MAX_AUDIO_CHUNK_BYTES = 24_000_000
DEFAULT_CHUNK_SECONDS = 30 * 60
DEFAULT_OVERLAP_SECONDS = 2.0
AUDIO_BITRATE = "32k"


class AudioProcessingError(RuntimeError):
    """Error de audio que se puede mostrar al usuario."""


@dataclass(frozen=True)
class AudioChunk:
    path: Path
    start: float
    end: float


def plan_audio_chunks(
    duration: float,
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
    overlap_seconds: float = DEFAULT_OVERLAP_SECONDS,
) -> list[tuple[float, float]]:
    if duration <= 0:
        raise ValueError("La duración del vídeo debe ser mayor que cero.")
    if chunk_seconds <= 0:
        raise ValueError("La duración de cada fragmento debe ser mayor que cero.")
    if overlap_seconds < 0 or overlap_seconds >= chunk_seconds:
        raise ValueError("El solapamiento debe ser menor que la duración del fragmento.")

    ranges: list[tuple[float, float]] = []
    boundary = 0.0
    while boundary < duration:
        start = max(0.0, boundary - overlap_seconds)
        end = min(duration, boundary + chunk_seconds)
        ranges.append((start, end))
        boundary += chunk_seconds
    return ranges


def create_audio_chunks(
    video_path: Path, work_dir: Path, config: AppConfig
) -> list[AudioChunk]:
    ffmpeg = _find_executable(config.ffmpeg_path)
    ffprobe = _find_ffprobe(ffmpeg)
    work_dir.mkdir(parents=True, exist_ok=True)
    duration = _probe_duration(ffprobe, video_path)
    compressed_audio = work_dir / "audio-16k-mono.mp3"
    _run(
        [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-b:a",
            AUDIO_BITRATE,
            "-codec:a",
            "libmp3lame",
            str(compressed_audio),
        ],
        "No se pudo extraer el audio. Comprueba que el vídeo contiene sonido.",
    )

    chunks: list[AudioChunk] = []
    for index, (start, end) in enumerate(plan_audio_chunks(duration)):
        chunk_path = work_dir / f"audio-{index:04d}.mp3"
        _run(
            [
                ffmpeg,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{start:.3f}",
                "-i",
                str(compressed_audio),
                "-t",
                f"{end - start:.3f}",
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-b:a",
                AUDIO_BITRATE,
                "-codec:a",
                "libmp3lame",
                str(chunk_path),
            ],
            "No se pudo dividir el audio para enviarlo a transcripción.",
        )
        if chunk_path.stat().st_size >= MAX_AUDIO_CHUNK_BYTES:
            chunk_path.unlink(missing_ok=True)
            raise AudioProcessingError(
                "Un fragmento de audio supera el límite seguro de 25 MB."
            )
        chunks.append(AudioChunk(chunk_path, start, end))

    compressed_audio.unlink(missing_ok=True)
    return chunks


def _find_executable(value: str) -> str:
    executable = find_ffmpeg(value)
    if executable:
        return executable
    raise AudioProcessingError(
        "No se encontró FFmpeg ni en el paquete de AiClip ni en la ruta "
        "configurada. Reinstala AiClip o comprueba ffmpeg_path en config.local.json."
    )


def _find_ffprobe(ffmpeg: str) -> str:
    sibling = Path(ffmpeg).with_name("ffprobe.exe" if ffmpeg.lower().endswith(".exe") else "ffprobe")
    if sibling.is_file():
        return str(sibling)
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        return ffprobe
    raise AudioProcessingError(
        "No se encontró FFprobe junto a FFmpeg. Comprueba la instalación."
    )


def _probe_duration(ffprobe: str, path: Path) -> float:
    try:
        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        duration = float(json.loads(result.stdout)["format"]["duration"])
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise AudioProcessingError(
            "No se pudo leer la duración del vídeo. Comprueba el archivo."
        ) from exc
    if duration <= 0:
        raise AudioProcessingError("El vídeo no tiene una duración válida.")
    return duration


def _run(command: list[str], error_message: str) -> None:
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=60 * 60,
        )
    except FileNotFoundError as exc:
        raise AudioProcessingError(
            "No se encontró FFmpeg. Instálalo o indica su ruta en config.local.json."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise AudioProcessingError(
            "El procesamiento de audio tardó demasiado y se canceló."
        ) from exc
    except subprocess.CalledProcessError as exc:
        if "no space left" in (exc.stderr or "").casefold():
            raise AudioProcessingError(
                "El disco está lleno. Libera espacio y vuelve a intentarlo."
            ) from exc
        raise AudioProcessingError(error_message) from exc
