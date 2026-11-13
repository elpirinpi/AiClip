import errno
import logging
from pathlib import Path
from urllib.parse import urlparse

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError


logger = logging.getLogger(__name__)


class IngestError(RuntimeError):
    """Error de entrada de vídeo que se puede mostrar al usuario."""


def resolve_video_source(source: str, work_dir: Path) -> Path:
    path = Path(source).expanduser()
    if path.is_file():
        return path.resolve()

    normalized_source = normalize_youtube_url(source)
    if normalized_source is None:
        raise IngestError(
            "Selecciona un archivo de vídeo existente o pega un enlace de YouTube válido."
        )

    work_dir.mkdir(parents=True, exist_ok=True)
    options = {
        "format": "best[ext=mp4]/best",
        "outtmpl": str(work_dir / "source.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "overwrites": True,
    }
    try:
        with YoutubeDL(options) as downloader:
            info = downloader.extract_info(normalized_source, download=True)
            filename = Path(downloader.prepare_filename(info))
    except DownloadError as exc:
        raise IngestError(_youtube_download_error_message(exc)) from exc
    except OSError as exc:
        winerror = getattr(exc, "winerror", None)
        logger.error(
            "Error de sistema al descargar el vídeo (tipo=%s, errno=%s, winerror=%s).",
            type(exc).__name__,
            exc.errno,
            winerror,
        )
        raise IngestError(_youtube_os_error_message(exc)) from exc

    if not filename.is_file():
        candidates = sorted(work_dir.glob("source.*"))
        if len(candidates) != 1:
            raise IngestError("YouTube no entregó un archivo de vídeo utilizable.")
        filename = candidates[0]
    return filename


def _youtube_download_error_message(error: DownloadError) -> str:
    details = str(error).casefold()
    if (
        "confirm you're not a bot" in details
        or "confirm you’re not a bot" in details
        or "http error 429" in details
        or "too many requests" in details
    ):
        return (
            "YouTube está limitando las solicitudes y pide verificar que no eres "
            "un bot (HTTP 429). Prueba más tarde o selecciona el vídeo como "
            "archivo local. AiClip no usa las cookies del navegador."
        )
    if "sign in" in details or "login_required" in details:
        return (
            "YouTube requiere iniciar sesión para ver este vídeo. AiClip no puede "
            "acceder a vídeos que requieren una cuenta; selecciona un archivo "
            "local si tienes derecho a procesarlo."
        )
    if "private video" in details or "this video is private" in details:
        return (
            "Este vídeo de YouTube es privado y no se puede descargar desde "
            "AiClip. Selecciona un archivo local si tienes derecho a procesarlo."
        )
    return (
        "No se pudo descargar el vídeo de YouTube. Comprueba que el enlace sea "
        "válido, que el vídeo esté disponible y vuelve a intentarlo."
    )


def _youtube_os_error_message(error: OSError) -> str:
    winerror = getattr(error, "winerror", None)
    error_code = winerror or error.errno
    if error.errno == errno.ENOSPC or winerror == 112:
        return "El disco está lleno. Libera espacio y vuelve a intentarlo."
    if winerror in {32, 33}:
        return (
            "Otro programa está usando un archivo de la descarga. Cierra ese "
            "programa y vuelve a intentarlo."
        )
    if error.errno in {errno.EACCES, errno.EPERM} or winerror == 5:
        return (
            "Windows no permite guardar el vídeo en la carpeta de trabajo. "
            "Comprueba los permisos de la carpeta y vuelve a intentarlo."
        )
    if winerror == 206:
        return (
            "La ruta de la carpeta de trabajo es demasiado larga. Cambia la "
            "carpeta de salida por una ruta más corta en Ajustes."
        )
    if error.errno == errno.ENOENT:
        return (
            "No se encontró un componente necesario para completar la descarga. "
            "Reinicia AiClip y vuelve a intentarlo."
        )
    if error_code in {10051, 10053, 10054, 10060, 10065, 11001}:
        return (
            "La conexión con YouTube se interrumpió durante la descarga. "
            "Comprueba Internet y vuelve a intentarlo más tarde."
        )
    code = f" (código {error_code})" if error_code is not None else ""
    return (
        "Windows no pudo guardar el vídeo descargado"
        f"{code}. Comprueba el espacio disponible y los permisos de la carpeta."
    )


def is_youtube_url(source: str) -> bool:
    return normalize_youtube_url(source) is not None


def normalize_youtube_url(source: str) -> str | None:
    candidate = source.strip()
    if not candidate:
        return None
    if "://" not in candidate:
        candidate = f"https://{candidate}"
    parsed = urlparse(candidate)
    try:
        port = parsed.port
    except ValueError:
        return None
    hostname = (parsed.hostname or "").casefold().rstrip(".")
    if (
        parsed.scheme.casefold() not in {"http", "https"}
        or parsed.username is not None
        or parsed.password is not None
        or port not in {None, 80, 443}
        or not (
            hostname == "youtu.be"
            or hostname == "youtube.com"
            or hostname.endswith(".youtube.com")
        )
    ):
        return None
    return parsed._replace(scheme="https", netloc=hostname).geturl()
