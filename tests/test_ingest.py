from pathlib import Path

import pytest
from yt_dlp.utils import DownloadError

import app.pipeline.ingest as ingest
from app.pipeline.ingest import (
    IngestError,
    is_youtube_url,
    normalize_youtube_url,
    resolve_video_source,
)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "https://www.youtube.com/watch?v=abc123",
            "https://www.youtube.com/watch?v=abc123",
        ),
        ("youtu.be/abc123?t=5", "https://youtu.be/abc123?t=5"),
        (
            "http://m.youtube.com/shorts/abc123",
            "https://m.youtube.com/shorts/abc123",
        ),
        (
            "https://youtube.com/live/abc123",
            "https://youtube.com/live/abc123",
        ),
    ],
)
def test_youtube_links_are_normalized(
    source: str,
    expected: str,
) -> None:
    assert normalize_youtube_url(source) == expected
    assert is_youtube_url(source)


@pytest.mark.parametrize(
    "source",
    [
        "",
        "https://youtube.com.example.org/watch?v=abc123",
        "https://example.com/watch?v=abc123",
        "https://user:password@youtube.com/watch?v=abc123",
        "https://youtube.com:8443/watch?v=abc123",
    ],
)
def test_non_youtube_or_unsafe_links_are_rejected(source: str) -> None:
    assert normalize_youtube_url(source) is None
    assert not is_youtube_url(source)


@pytest.mark.parametrize(
    ("download_error", "expected_message"),
    [
        (
            "HTTP Error 429: Too Many Requests; Sign in to confirm you're not a bot",
            "YouTube está limitando las solicitudes",
        ),
        (
            "LOGIN_REQUIRED: Sign in to view this video",
            "YouTube requiere iniciar sesión",
        ),
        (
            "This video is private",
            "Este vídeo de YouTube es privado",
        ),
        (
            "Video unavailable",
            "Comprueba que el enlace sea válido",
        ),
    ],
)
def test_youtube_download_errors_explain_the_cause(
    monkeypatch,
    tmp_path: Path,
    download_error: str,
    expected_message: str,
) -> None:
    class FailingDownloader:
        def __init__(self, _options):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def extract_info(self, _source, download):
            assert download is True
            raise DownloadError(download_error)

    monkeypatch.setattr(ingest, "YoutubeDL", FailingDownloader)

    with pytest.raises(IngestError, match=expected_message):
        resolve_video_source(
            "https://www.youtube.com/watch?v=Rcjf56X7Zvg",
            tmp_path / "working",
        )


@pytest.mark.parametrize(
    ("error", "expected_message"),
    [
        (OSError(28, "disk full"), "El disco está lleno"),
        (OSError(13, "permission denied"), "Comprueba los permisos"),
        (OSError(10054, "connection reset"), "La conexión con YouTube"),
        (OSError(5, "access denied", None, 5), "Comprueba los permisos"),
        (OSError(32, "file in use", None, 32), "Otro programa está usando"),
        (OSError(206, "path too long", None, 206), "ruta de la carpeta"),
    ],
)
def test_youtube_filesystem_errors_explain_how_to_recover(
    monkeypatch,
    tmp_path: Path,
    error: OSError,
    expected_message: str,
) -> None:
    class FailingDownloader:
        def __init__(self, _options):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def extract_info(self, _source, download):
            assert download is True
            raise error

    monkeypatch.setattr(ingest, "YoutubeDL", FailingDownloader)

    with pytest.raises(IngestError, match=expected_message):
        resolve_video_source(
            "https://www.youtube.com/watch?v=Rcjf56X7Zvg",
            tmp_path / "working",
        )
