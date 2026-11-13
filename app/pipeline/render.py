import logging
import queue
import subprocess
import tempfile
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Callable

from app.config import AppConfig
from app.pipeline.ffmpeg import find_ffmpeg
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
    progress_callback: Callable[[float], None] | None = None,
) -> Path:
    ffmpeg = _find_executable(config.ffmpeg_path)
    work_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subtitle_path = work_dir / f"{output_path.stem}.ass"
    write_ass_subtitles(subtitle_path, clip, words, style)
    duration = clip.end - clip.start
    if duration <= 0:
        raise RenderError("El intervalo del clip no es válido.")
    if progress_callback is not None:
        progress_callback(0.05)
    if reframe_mode == "face":
        plan = analyze_face_tracking(video_path, clip.start, duration, ffmpeg)
    elif reframe_mode in ("center", "blur"):
        width, height = _video_dimensions(video_path)
        plan = build_reframe_plan(width, height, reframe_mode)
    else:
        raise RenderError("El modo de encuadre no es válido.")
    if progress_callback is not None:
        progress_callback(0.15)
    if plan.fallback_reason:
        logger.warning(plan.fallback_reason)

    filter_option, video_filter, video_map = _video_filter(plan, subtitle_path.name)
    filter_script = work_dir / f"{output_path.stem}.ffgraph"
    if filter_option == "-vf":
        filter_script.write_text(
            f"[0:v]{video_filter}[outv]\n",
            encoding="utf-8",
        )
    else:
        filter_script.write_text(f"{video_filter}\n", encoding="utf-8")
    if filter_option == "-vf":
        video_map = "[outv]"
    encoder = _select_video_encoder(ffmpeg)
    command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-stats_period",
        "0.5",
        "-progress",
        "pipe:1",
        "-y",
        "-ss",
        f"{clip.start:.3f}",
        "-i",
        str(video_path.resolve()),
        "-t",
        f"{duration:.3f}",
    ]
    command.extend(["-map", video_map or "0:v:0", "-map", "0:a?"])
    command.extend(["-filter_complex_script", filter_script.name])
    command.extend([
        "-c:v",
        encoder,
    ])
    if encoder == "h264_nvenc":
        command.extend(["-preset", "p5", "-cq", "20", "-b:v", "0"])
    else:
        command.extend(["-preset", "veryfast", "-crf", "21"])
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
        _run_ffmpeg_with_progress(
            command,
            work_dir,
            duration,
            progress_callback,
        )
    except FileNotFoundError as exc:
        logger.exception(
            "Could not start FFmpeg (executable=%r, exists=%s, cwd=%r, cwd_exists=%s)",
            ffmpeg,
            Path(ffmpeg).is_file(),
            str(work_dir),
            work_dir.is_dir(),
        )
        windows_error = (
            f" (error de Windows {exc.winerror})" if exc.winerror is not None else ""
        )
        raise RenderError(
            "Windows no pudo iniciar FFmpeg"
            f"{windows_error}. Comprueba que el ejecutable y la carpeta de trabajo "
            "estén disponibles."
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


def _run_ffmpeg_with_progress(
    command: list[str],
    work_dir: Path,
    duration: float,
    progress_callback: Callable[[float], None] | None,
) -> None:
    progress_lines: queue.Queue[str | None] = queue.Queue()
    timeout = 60 * 60
    started_at = time.monotonic()
    process: subprocess.Popen[str] | None = None

    try:
        with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as stderr_file:
            process = subprocess.Popen(
                command,
                cwd=work_dir,
                stdout=subprocess.PIPE,
                stderr=stderr_file,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            if process.stdout is None:
                raise RenderError("No se pudo leer el progreso de FFmpeg.")

            def read_progress() -> None:
                try:
                    for line in process.stdout:
                        progress_lines.put(line)
                finally:
                    progress_lines.put(None)

            reader = threading.Thread(target=read_progress, daemon=True)
            reader.start()
            reader_finished = False
            current_output_time = 0.0
            while process.poll() is None or not reader_finished:
                remaining = timeout - (time.monotonic() - started_at)
                if remaining <= 0:
                    process.kill()
                    process.wait()
                    raise subprocess.TimeoutExpired(command, timeout)
                try:
                    line = progress_lines.get(timeout=min(0.25, remaining))
                except queue.Empty:
                    continue
                if line is None:
                    reader_finished = True
                    continue
                key, separator, value = line.strip().partition("=")
                if not separator:
                    continue
                if key == "out_time_us":
                    current_output_time = int(value) / 1_000_000
                elif key == "out_time":
                    hours, minutes, seconds = value.split(":")
                    current_output_time = (
                        int(hours) * 3600 + int(minutes) * 60 + float(seconds)
                    )
                elif key == "progress" and progress_callback is not None:
                    progress_callback(
                        min(0.99, max(0.0, current_output_time / duration))
                    )

            return_code = process.wait()
            stderr_file.seek(0)
            stderr = stderr_file.read()
            if return_code:
                raise subprocess.CalledProcessError(
                    return_code,
                    command,
                    stderr=stderr,
                )
    except BaseException:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        raise


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

    small_width = OUTPUT_WIDTH // 8
    small_height = OUTPUT_HEIGHT // 8
    graph = (
        f"[0:v]split=2[bg][fg];"
        f"[bg]scale={small_width}:{small_height}:force_original_aspect_ratio=increase,"
        f"crop={small_width}:{small_height},boxblur=4:2,"
        f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}[blurred];"
        f"[fg]scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:force_original_aspect_ratio=decrease[fit];"
        f"[blurred][fit]overlay=(W-w)/2:(H-h)/2,{subtitle_filter}[outv]"
    )
    return "-filter_complex", graph, "[outv]"


@lru_cache(maxsize=4)
def _select_video_encoder(ffmpeg: str) -> str:
    return "h264_nvenc" if probe_nvenc(ffmpeg) else "libx264"


def _find_executable(value: str) -> str:
    executable = find_ffmpeg(value)
    if executable:
        return executable
    raise RenderError(
        "No se encontró FFmpeg ni en el paquete de AiClip ni en la ruta "
        "configurada. Reinstala AiClip o comprueba ffmpeg_path en config.local.json."
    )
