import logging
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

from app.config import AppConfig
from app.pipeline.reframe import (
    OUTPUT_HEIGHT,
    OUTPUT_WIDTH,
    ReframeError,
    ReframePlan,
    analyze_face_tracking,
    build_crop_filter,
    build_reframe_plan,
    probe_nvenc,
)
from app.pipeline.segment import Clip
from app.pipeline.subtitles import write_ass_subtitles
from app.providers.base import TranscriptWord


logger = logging.getLogger(__name__)


class RenderError(RuntimeError):
    """Error legible al renderizar un clip."""


def render_clip(
    video_path: Path,
    clip: Clip,
    words: list[TranscriptWord],
    output_path: Path,
    work_dir: Path,
    style: str,
    config: AppConfig,
    reframe_mode: str = "center",
) -> Path:
    ffmpeg = _find_executable(config.ffmpeg_path)
    work_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subtitle_path = work_dir / f"{output_path.stem}.ass"
    write_ass_subtitles(subtitle_path, clip, words, style)
    duration = clip.end - clip.start
    if duration <= 0:
        raise RenderError("El intervalo del clip no es válido.")
    if reframe_mode == "face":
        plan = analyze_face_tracking(video_path, clip.start, duration)
    elif reframe_mode in ("center", "blur"):
        width, height = _video_dimensions(video_path)
        plan = build_reframe_plan(width, height, reframe_mode)
    else:
        raise RenderError("El modo de encuadre no es válido.")
    if plan.fallback_reason:
        logger.warning(plan.fallback_reason)

    filter_option, video_filter, video_map = _video_filter(plan, subtitle_path.name)
    encoder = _select_video_encoder(ffmpeg)
    command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{clip.start:.3f}",
        "-i",
        str(video_path.resolve()),
        "-t",
        f"{duration:.3f}",
    ]
    command.extend(["-map", video_map or "0:v:0", "-map", "0:a?"])
    command.extend([filter_option, video_filter])
    command.extend([
        "-c:v",
        encoder,
    ])
    if encoder == "h264_nvenc":
        command.extend(["-preset", "p5", "-cq", "20", "-b:v", "0"])
    else:
        command.extend(["-preset", "medium", "-crf", "20"])
    command.extend([
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        str(output_path.resolve()),
    ])
    try:
        subprocess.run(
            command,
            cwd=work_dir,
            check=True,
            capture_output=True,
            text=True,
            timeout=60 * 60,
        )
    except FileNotFoundError as exc:
        raise RenderError(
            "No se encontró FFmpeg. Instálalo o indica su ruta en config.local.json."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RenderError("El renderizado tardó demasiado y se canceló.") from exc
    except subprocess.CalledProcessError as exc:
        output_path.unlink(missing_ok=True)
        details = (exc.stderr or "").casefold()
        if "no space left" in details or "disk full" in details:
            raise RenderError(
                "El disco está lleno. Libera espacio y vuelve a intentarlo."
            ) from exc
        raise RenderError(
            "FFmpeg no pudo cortar el clip o quemar los subtítulos. "
            "Comprueba el vídeo y la instalación de FFmpeg."
        ) from exc
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RenderError("FFmpeg no creó el archivo de vídeo esperado.")
    return output_path


def _video_dimensions(path: Path) -> tuple[int, int]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RenderError("No se pudo abrir el vídeo para el recorte vertical.")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            raise RenderError("No se pudieron leer las dimensiones del vídeo.")
        return width, height
    finally:
        capture.release()


def _video_filter(plan: ReframePlan, subtitle_name: str) -> tuple[str, str, str | None]:
    subtitle_filter = f"subtitles=filename='{subtitle_name}'"
    if plan.mode != "blur":
        return "-vf", f"{build_crop_filter(plan)},{subtitle_filter}", None

    graph = (
        f"[0:v]split=2[bg][fg];"
        f"[bg]scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:force_original_aspect_ratio=increase,"
        f"crop={OUTPUT_WIDTH}:{OUTPUT_HEIGHT},boxblur=20:10[blurred];"
        f"[fg]scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:force_original_aspect_ratio=decrease[fit];"
        f"[blurred][fit]overlay=(W-w)/2:(H-h)/2,{subtitle_filter}[outv]"
    )
    return "-filter_complex", graph, "[outv]"


@lru_cache(maxsize=4)
def _select_video_encoder(ffmpeg: str) -> str:
    return "h264_nvenc" if probe_nvenc(ffmpeg) else "libx264"


def _find_executable(value: str) -> str:
    executable = shutil.which(value)
    if executable:
        return executable
    path = Path(value)
    if path.is_file():
        return str(path)
    raise RenderError(
        "No se encontró FFmpeg. Instálalo o indica su ruta en config.local.json."
    )
