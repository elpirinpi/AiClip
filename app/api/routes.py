import asyncio
import json
import logging
import shutil
import sqlite3
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.config import AppConfig, ConfigError, load_config, save_settings
from app.db import (
    DEFAULT_DATABASE_PATH,
    DatabaseError,
    add_manual_clip,
    create_job,
    get_clip,
    get_clips,
    get_job,
    get_job_clips,
    get_transcript_duration,
    get_transcript_words,
    list_jobs,
    list_queued_jobs,
    save_results,
    set_clip_selection,
    update_clip,
    update_job,
)
from app.pipeline.analyze import AnalysisError, analyze_transcript
from app.pipeline.audio import AudioProcessingError, create_audio_chunks
from app.pipeline.ingest import IngestError, is_youtube_url, resolve_video_source
from app.pipeline.render import RenderError, render_clip
from app.pipeline.segment import Clip
from app.pipeline.thumbnail import ThumbnailError, create_thumbnail
from app.pipeline.transcribe import transcribe_chunks
from app.providers.base import ProviderError
from app.providers.gemini_llm import create_analyzer
from app.providers.groq_stt import create_transcriber


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")
job_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="aiclip-job")
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024 * 1024


class SettingsUpdate(BaseModel):
    groq_api_key: str | None = None
    gemini_api_key: str | None = None
    output_folder: str | None = None
    subtitle_style: Literal["classic", "impact"] | None = None
    reframe_mode: Literal["center", "face", "blur"] | None = None


class ClipUpdate(BaseModel):
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    title: str = Field(min_length=1, max_length=120)
    selected: bool


class ManualClipRequest(BaseModel):
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    title: str = Field(min_length=1, max_length=120)


class SelectionUpdate(BaseModel):
    selected: bool


def _job_config() -> AppConfig:
    config = load_config()
    if config.transcription_fallback or config.analysis_fallback:
        raise ProviderError("El proveedor de fallback configurado no está disponible.")
    return config


def _job_row(job_id: str):
    job = get_job(job_id, DEFAULT_DATABASE_PATH)
    if job is None:
        raise HTTPException(status_code=404, detail="No se encontró el trabajo.")
    return job


def _serialize_job(job) -> dict[str, object]:
    return {
        "id": job["id"],
        "source": job["source"],
        "status": job["status"],
        "progress": job["progress"],
        "error": job["error"],
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
    }


def _serialize_clip(clip) -> dict[str, object]:
    return {
        "id": clip["id"],
        "start": clip["start"],
        "end": clip["end"],
        "title": clip["title"],
        "reason": clip["reason"],
        "score": clip["score"],
        "selected": bool(clip["selected"]),
    }


def _schedule_job(job_id: str, operation: str) -> None:
    task = _process_job if operation == "process" else _render_job
    job_executor.submit(task, job_id)


def _process_job(job_id: str) -> None:
    try:
        job = get_job(job_id, DEFAULT_DATABASE_PATH)
        if job is None:
            return
        config = _job_config()
        job_dir = config.output_folder / "jobs" / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        update_job(job_id, status="processing", progress=5, path=DEFAULT_DATABASE_PATH)
        input_path = Path(job["input_path"]) if job["input_path"] else None
        if input_path is None or not input_path.is_file():
            video_path = resolve_video_source(job["source"], job_dir / "source")
            update_job(
                job_id,
                input_path=str(video_path),
                progress=20,
                path=DEFAULT_DATABASE_PATH,
            )
        else:
            video_path = input_path
            update_job(job_id, progress=20, path=DEFAULT_DATABASE_PATH)

        chunks = create_audio_chunks(video_path, job_dir / "working" / "audio", config)
        update_job(job_id, progress=40, path=DEFAULT_DATABASE_PATH)
        words = transcribe_chunks(chunks, create_transcriber(config))
        if not words:
            raise ProviderError(
                "Groq no reconoció palabras en el audio. Comprueba el sonido del vídeo."
            )
        update_job(job_id, progress=70, path=DEFAULT_DATABASE_PATH)
        analyzer = create_analyzer(config)
        try:
            candidates = analyze_transcript(words, analyzer)
        finally:
            analyzer.close()
        from app.pipeline.segment import snap_clips_to_sentences

        clips, warnings = snap_clips_to_sentences(candidates, words)
        save_results(job_id, clips, words, DEFAULT_DATABASE_PATH)
        update_job(job_id, status="ready", progress=100, path=DEFAULT_DATABASE_PATH)
        for warning in warnings:
            logger.info("Trabajo %s: %s", job_id, warning)
    except (
        ConfigError,
        DatabaseError,
        IngestError,
        AudioProcessingError,
        AnalysisError,
        ProviderError,
        OSError,
    ) as exc:
        _fail_job(job_id, str(exc))
    except Exception:
        logger.exception("Error inesperado en el trabajo %s.", job_id)
        _fail_job(job_id, "Ocurrió un error inesperado. Inténtalo de nuevo.")


def _render_job(job_id: str) -> None:
    analyzer = None
    try:
        job = get_job(job_id, DEFAULT_DATABASE_PATH)
        if job is None:
            return
        config = _job_config()
        job_dir = config.output_folder / "jobs" / job_id
        video_path = Path(job["input_path"])
        if not video_path.is_file():
            raise IngestError("No se encontró el vídeo del trabajo.")
        clips = get_job_clips(job_id, DEFAULT_DATABASE_PATH)
        selected = [clip for clip in clips if clip["selected"]]
        if not selected:
            raise RenderError("Selecciona al menos un clip para exportar.")
        words = get_transcript_words(job_id, DEFAULT_DATABASE_PATH)
        if not words:
            raise DatabaseError("No se encontró la transcripción del trabajo.")
        output_dir = config.output_folder / "clips"
        thumbnail_dir = config.output_folder / "thumbnails"
        output_dir.mkdir(parents=True, exist_ok=True)
        thumbnail_dir.mkdir(parents=True, exist_ok=True)
        analyzer = create_analyzer(config)
        update_job(job_id, status="rendering", progress=5, path=DEFAULT_DATABASE_PATH)
        for index, row in enumerate(selected, start=1):
            clip = Clip(
                start=row["start"],
                end=row["end"],
                title=row["title"],
                reason=row["reason"],
                score=row["score"],
            )
            create_thumbnail(
                video_path,
                clip,
                words,
                analyzer,
                thumbnail_dir / f"clip_{row['id']:05d}.png",
            )
            output_path = output_dir / (
                f"clip_{row['id']:05d}_{config.reframe_mode}_{config.subtitle_style}.mp4"
            )
            render_clip(
                video_path,
                clip,
                words,
                output_path,
                job_dir / "render_work",
                config.subtitle_style,
                config,
                reframe_mode=config.reframe_mode,
            )
            progress = round(5 + 90 * index / len(selected))
            update_job(job_id, status="rendering", progress=progress, path=DEFAULT_DATABASE_PATH)
        update_job(job_id, status="completed", progress=100, path=DEFAULT_DATABASE_PATH)
    except (
        ConfigError,
        DatabaseError,
        IngestError,
        RenderError,
        ThumbnailError,
        ProviderError,
        OSError,
    ) as exc:
        _fail_job(job_id, str(exc))
    except Exception:
        logger.exception("Error inesperado al exportar el trabajo %s.", job_id)
        _fail_job(job_id, "Ocurrió un error inesperado al exportar.")
    finally:
        if analyzer is not None:
            analyzer.close()


def _fail_job(job_id: str, error: str) -> None:
    try:
        update_job(
            job_id,
            status="failed",
            error=error,
            path=DEFAULT_DATABASE_PATH,
        )
    except DatabaseError:
        logger.exception("No se pudo guardar el error del trabajo %s.", job_id)


def recover_jobs() -> None:
    for job in list_jobs(DEFAULT_DATABASE_PATH):
        if job["status"] in {"processing", "rendering"}:
            update_job(
                job["id"],
                status="failed",
                error="La aplicación se cerró mientras este trabajo estaba en curso.",
                path=DEFAULT_DATABASE_PATH,
            )
    for job in list_queued_jobs(DEFAULT_DATABASE_PATH):
        _schedule_job(
            job["id"],
            "render" if job["status"] == "render_queued" else "process",
        )


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/settings")
def read_settings() -> dict[str, object]:
    try:
        config = load_config()
    except ConfigError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "groq_key_configured": bool(config.groq_api_key),
        "gemini_key_configured": bool(config.gemini_api_key),
        "output_folder": str(config.output_folder),
        "subtitle_style": config.subtitle_style,
        "reframe_mode": config.reframe_mode,
    }


@router.put("/settings")
def write_settings(settings: SettingsUpdate) -> dict[str, object]:
    try:
        config = save_settings(
            groq_api_key=settings.groq_api_key,
            gemini_api_key=settings.gemini_api_key,
            output_folder=settings.output_folder,
            subtitle_style=settings.subtitle_style,
            reframe_mode=settings.reframe_mode,
        )
    except ConfigError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "groq_key_configured": bool(config.groq_api_key),
        "gemini_key_configured": bool(config.gemini_api_key),
        "output_folder": str(config.output_folder),
        "subtitle_style": config.subtitle_style,
        "reframe_mode": config.reframe_mode,
    }


@router.get("/jobs")
def read_jobs() -> list[dict[str, object]]:
    try:
        return [_serialize_job(job) for job in list_jobs(DEFAULT_DATABASE_PATH)]
    except DatabaseError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/jobs", status_code=202)
async def submit_job(
    source: str = Form(default=""),
    video: UploadFile | None = File(default=None),
) -> dict[str, object]:
    source = source.strip()
    if (video is None) == (not source):
        raise HTTPException(
            status_code=400,
            detail="Pega un enlace de YouTube o selecciona un archivo de vídeo.",
        )
    job_id = uuid.uuid4().hex
    config = _load_settings_or_http_error()
    job_dir = config.output_folder / "jobs" / job_id
    input_path = ""
    try:
        job_dir.mkdir(parents=True, exist_ok=True)
        if video is not None:
            suffix = Path(video.filename or "").suffix.casefold()
            if suffix not in VIDEO_EXTENSIONS:
                raise HTTPException(
                    status_code=400,
                    detail="El archivo debe ser un vídeo MP4, MOV, MKV, WEBM, AVI o M4V.",
                )
            input_path = str(job_dir / f"input{suffix}")
            total = 0
            with Path(input_path).open("wb") as target:
                while chunk := await video.read(1024 * 1024):
                    total += len(chunk)
                    if total > MAX_UPLOAD_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail="El archivo supera el límite de 20 GB.",
                        )
                    target.write(chunk)
            if total == 0:
                raise HTTPException(status_code=400, detail="El archivo está vacío.")
            label = Path(video.filename or "video").name
        else:
            if not is_youtube_url(source):
                raise HTTPException(
                    status_code=400,
                    detail="Pega un enlace HTTPS válido de YouTube.",
                )
            label = source

        create_job(job_id, label, input_path, DEFAULT_DATABASE_PATH)
        _schedule_job(job_id, "process")
        return _serialize_job(get_job(job_id, DEFAULT_DATABASE_PATH))
    except HTTPException:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
    except (OSError, DatabaseError) as exc:
        shutil.rmtree(job_dir, ignore_errors=True)
        if isinstance(exc, OSError) and getattr(exc, "errno", None) == 28:
            raise HTTPException(
                status_code=507,
                detail="El disco está lleno. Libera espacio y vuelve a intentarlo.",
            ) from exc
        if isinstance(exc, DatabaseError):
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        raise HTTPException(status_code=500, detail="No se pudo guardar el vídeo.") from exc
    finally:
        if video is not None:
            await video.close()


@router.get("/jobs/{job_id}")
def read_job(job_id: str) -> dict[str, object]:
    job = _job_row(job_id)
    try:
        response = _serialize_job(job)
        response["clips"] = [
            _serialize_clip(clip)
            for clip in get_job_clips(job_id, DEFAULT_DATABASE_PATH)
        ]
        return response
    except DatabaseError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.put("/jobs/{job_id}/clips/{clip_id}")
def edit_clip(job_id: str, clip_id: int, update: ClipUpdate) -> dict[str, object]:
    _require_ready_job(job_id)
    duration = get_transcript_duration(job_id, DEFAULT_DATABASE_PATH)
    if (
        not update.start < update.end
        or update.end - update.start < 1
        or update.end > duration + 0.1
    ):
        raise HTTPException(
            status_code=400,
            detail="El rango del clip debe ser válido y estar dentro del vídeo.",
        )
    if not update.title.strip():
        raise HTTPException(status_code=400, detail="El título no puede estar vacío.")
    try:
        update_clip(
            job_id,
            clip_id,
            start=update.start,
            end=update.end,
            title=update.title.strip(),
            selected=update.selected,
            path=DEFAULT_DATABASE_PATH,
        )
        clip = get_clip(job_id, clip_id, DEFAULT_DATABASE_PATH)
        if clip is None:
            raise HTTPException(status_code=404, detail="No se encontró el clip.")
        return _serialize_clip(clip)
    except DatabaseError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/jobs/{job_id}/clips")
def create_manual_clip(
    job_id: str, request: ManualClipRequest
) -> dict[str, object]:
    _require_ready_job(job_id)
    duration = get_transcript_duration(job_id, DEFAULT_DATABASE_PATH)
    if (
        not request.start < request.end
        or request.end - request.start < 1
        or request.end > duration + 0.1
    ):
        raise HTTPException(
            status_code=400,
            detail="El rango del clip debe ser válido y estar dentro del vídeo.",
        )
    try:
        clip_id = add_manual_clip(
            job_id,
            request.start,
            request.end,
            request.title.strip(),
            DEFAULT_DATABASE_PATH,
        )
        clip = get_clip(job_id, clip_id, DEFAULT_DATABASE_PATH)
        return _serialize_clip(clip)
    except DatabaseError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.put("/jobs/{job_id}/selection")
def update_selection(job_id: str, update: SelectionUpdate) -> dict[str, str]:
    _require_ready_job(job_id)
    try:
        set_clip_selection(
            job_id, update.selected, path=DEFAULT_DATABASE_PATH
        )
        return {"status": "ok"}
    except DatabaseError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/jobs/{job_id}/render", status_code=202)
def render_selected(job_id: str) -> dict[str, object]:
    _require_ready_job(job_id)
    try:
        selected = [
            clip
            for clip in get_job_clips(job_id, DEFAULT_DATABASE_PATH)
            if clip["selected"]
        ]
        if not selected:
            raise HTTPException(
                status_code=400,
                detail="Selecciona al menos un clip para exportar.",
            )
        update_job(
            job_id,
            status="render_queued",
            progress=0,
            error=None,
            path=DEFAULT_DATABASE_PATH,
        )
        _schedule_job(job_id, "render")
        return _serialize_job(get_job(job_id, DEFAULT_DATABASE_PATH))
    except DatabaseError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/jobs/{job_id}/clips/{clip_id}/preview")
def preview_clip(
    job_id: str,
    clip_id: int,
    style: Literal["classic", "impact"] | None = None,
    reframe: Literal["center", "face", "blur"] | None = None,
) -> FileResponse:
    job = _require_job_status(job_id, {"completed"})
    clip = get_clip(job_id, clip_id, DEFAULT_DATABASE_PATH)
    if clip is None or not clip["selected"]:
        raise HTTPException(status_code=404, detail="No se encontró el clip exportado.")
    config = _load_settings_or_http_error()
    output = (
        config.output_folder
        / "clips"
        / f"clip_{clip_id:05d}_{reframe or config.reframe_mode}_"
        f"{style or config.subtitle_style}.mp4"
    )
    if not output.is_file():
        raise HTTPException(status_code=404, detail="Todavía no existe una vista previa.")
    return FileResponse(output, media_type="video/mp4")


@router.get("/jobs/{job_id}/download")
def download_selected(job_id: str) -> FileResponse:
    _require_job_status(job_id, {"completed"})
    config = _load_settings_or_http_error()
    clips = [
        clip
        for clip in get_job_clips(job_id, DEFAULT_DATABASE_PATH)
        if clip["selected"]
    ]
    if not clips:
        raise HTTPException(status_code=404, detail="No hay clips seleccionados.")
    job_dir = config.output_folder / "jobs" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    archive_path = job_dir / "clips-seleccionados.zip"
    try:
        with zipfile.ZipFile(
            archive_path, mode="w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            for clip in clips:
                render_path = config.output_folder / "clips" / (
                    f"clip_{clip['id']:05d}_{config.reframe_mode}_"
                    f"{config.subtitle_style}.mp4"
                )
                thumbnail_path = config.output_folder / "thumbnails" / (
                    f"clip_{clip['id']:05d}.png"
                )
                if render_path.is_file():
                    archive.write(render_path, render_path.name)
                if thumbnail_path.is_file():
                    archive.write(thumbnail_path, thumbnail_path.name)
        if archive_path.stat().st_size == 0:
            raise HTTPException(
                status_code=404,
                detail="Los archivos aún no están listos para descargar.",
            )
    except OSError as exc:
        if getattr(exc, "errno", None) == 28:
            raise HTTPException(
                status_code=507,
                detail="El disco está lleno. Libera espacio y vuelve a intentarlo.",
            ) from exc
        raise HTTPException(
            status_code=500, detail="No se pudo preparar la descarga."
        ) from exc
    return FileResponse(
        archive_path,
        media_type="application/zip",
        filename=f"aiclip-{job_id[:8]}.zip",
    )


def _require_ready_job(job_id: str):
    return _require_job_status(job_id, {"ready", "completed"})


def _require_job_status(job_id: str, statuses: set[str]):
    job = _job_row(job_id)
    if job["status"] not in statuses:
        raise HTTPException(
            status_code=409,
            detail="Espera a que termine el trabajo antes de hacer esta acción.",
        )
    return job


def _load_settings_or_http_error() -> AppConfig:
    try:
        return load_config()
    except ConfigError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
