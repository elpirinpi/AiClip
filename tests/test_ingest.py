from pathlib import Path

import pytest
from yt_dlp.utils import DownloadError

import app.pipeline.ingest as ingest
from app.pipeline.ingest import IngestError, resolve_video_source


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
