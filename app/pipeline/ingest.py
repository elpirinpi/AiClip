from pathlib import Path
from urllib.parse import urlparse

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError


class IngestError(RuntimeError):
    """Error de entrada de vídeo que se puede mostrar al usuario."""


def resolve_video_source(source: str, work_dir: Path) -> Path:
    path = Path(source).expanduser()
    if path.is_file():
        return path.resolve()

    if not is_youtube_url(source):
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
            info = downloader.extract_info(source, download=True)
            filename = Path(downloader.prepare_filename(info))
    except DownloadError as exc:
        raise IngestError(
            "No se pudo descargar el vídeo de YouTube. Comprueba el enlace y su disponibilidad."
        ) from exc
    except OSError as exc:
        if getattr(exc, "errno", None) == 28:
            raise IngestError(
                "El disco está lleno. Libera espacio y vuelve a intentarlo."
            ) from exc
        raise IngestError("No se pudo guardar el vídeo descargado.") from exc

    if not filename.is_file():
        candidates = sorted(work_dir.glob("source.*"))
        if len(candidates) != 1:
            raise IngestError("YouTube no entregó un archivo de vídeo utilizable.")
        filename = candidates[0]
    return filename


def is_youtube_url(source: str) -> bool:
    parsed = urlparse(source)
    hostname = (parsed.hostname or "").lower()
    return parsed.scheme == "https" and (
        hostname == "youtu.be"
        or hostname.endswith(".youtube.com")
        or hostname == "youtube.com"
    )
