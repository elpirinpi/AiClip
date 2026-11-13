import io
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

import app.api.routes as routes
import app.config as config_module
import scripts.start_app as app_launcher
from app.db import create_job, get_job, save_clip_export, save_results, update_job
from app.main import app
from app.pipeline.segment import Clip
from app.providers.base import TranscriptWord


def _configure_test_paths(monkeypatch, tmp_path: Path) -> tuple[Path, Path]:
    config_path = tmp_path / "config.local.json"
    database_path = tmp_path / "clips.sqlite3"
    output_path = tmp_path / "output"
    config_path.write_text(
        json.dumps(
            {
                "groq_api_key": "existing-groq-secret",
                "gemini_api_key": "existing-gemini-secret",
                "ffmpeg_path": "ffmpeg",
                "output_folder": str(output_path),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", config_path)
    monkeypatch.setattr(routes, "DEFAULT_DATABASE_PATH", database_path)
    return config_path, database_path


def test_web_home_and_health_expose_consent_warning(tmp_path: Path, monkeypatch) -> None:
    _configure_test_paths(monkeypatch, tmp_path)

    with TestClient(app, client=("127.0.0.1", 12345)) as client:
        home = client.get("/")
        health = client.get("/api/health")

    assert home.status_code == 200
    assert "El audio y el texto de la transcripción se envían a Groq" in home.text
    assert "eres responsable de tener derecho a procesar el contenido" in home.text
    assert "Desarrollador: Kelly Benjamin de la Cruz" in home.text
    assert "Colaborador: Luis Felipe" in home.text
    assert 'id="clearHistory"' in home.text
    assert 'id="retryJob"' in home.text
    assert 'id="saveSourceVideo"' in home.text
    assert health.json() == {"status": "ok"}


def test_updates_endpoint_returns_release_check_result(monkeypatch) -> None:
    result = {
        "current_version": "0.1.0",
        "latest_version": "0.2.0",
        "update_available": True,
        "release_url": "https://github.com/elpirinpi/AiClip/releases/tag/v0.2.0",
        "download_url": "https://github.com/elpirinpi/AiClip/releases/download/v0.2.0/AiClip-Setup.msi",
        "installer_available": True,
    }
    monkeypatch.setattr(routes, "check_for_updates", lambda: result)

    with TestClient(app, client=("127.0.0.1", 12345)) as client:
        response = client.get("/api/updates")

    assert response.status_code == 200
    assert response.json() == result


def test_settings_api_never_returns_keys_and_preserves_blank_inputs(
    tmp_path: Path, monkeypatch
) -> None:
    config_path, _ = _configure_test_paths(monkeypatch, tmp_path)

    with TestClient(app, client=("127.0.0.1", 12345)) as client:
        current = client.get("/api/settings")
        updated = client.put(
            "/api/settings",
            json={
                "groq_api_key": "",
                "gemini_api_key": "",
                "subtitle_style": "impact",
                "reframe_mode": "blur",
            },
        )

    assert current.status_code == updated.status_code == 200
    assert "existing-groq-secret" not in current.text
    assert "existing-gemini-secret" not in current.text
    assert "gemini_key_configured" not in current.json()
    data = json.loads(config_path.read_text(encoding="utf-8"))
    assert data["groq_api_key"] == "existing-groq-secret"
    assert data["gemini_api_key"] == "existing-gemini-secret"
    assert updated.json()["subtitle_style"] == "impact"
    assert updated.json()["reframe_mode"] == "blur"


def test_clip_edit_selection_and_manual_clip_api(
    tmp_path: Path, monkeypatch
) -> None:
    _, database_path = _configure_test_paths(monkeypatch, tmp_path)
    job_id = "a" * 32
    create_job(job_id, "video.mp4", str(tmp_path / "video.mp4"), database_path)
    update_job(job_id, status="ready", path=database_path)
    words = [
        TranscriptWord(0, 1, "Opening."),
        TranscriptWord(5, 6, "A"),
        TranscriptWord(10, 11, "moment."),
        TranscriptWord(35, 36, "Closing."),
    ]
    save_results(
        job_id,
        [Clip(5, 35, "Suggested moment", "A reason.", 8)],
        words,
        database_path,
    )
    monkeypatch.setattr(routes, "_schedule_job", lambda *args: None)

    with TestClient(app, client=("127.0.0.1", 12345)) as client:
        clips = client.get(f"/api/jobs/{job_id}").json()["clips"]
        edited = client.put(
            f"/api/jobs/{job_id}/clips/{clips[0]['id']}",
            json={
                "start": 5.2,
                "end": 35,
                "title": "Edited moment",
                "selected": False,
            },
        )
        selected = client.put(
            f"/api/jobs/{job_id}/selection", json={"selected": True}
        )
        manual = client.post(
            f"/api/jobs/{job_id}/clips",
            json={"start": 1, "end": 32, "title": "Manual moment"},
        )

    assert edited.status_code == 200
    assert edited.json()["title"] == "Edited moment"
    assert edited.json()["selected"] is False
    assert selected.json() == {"status": "ok"}
    assert manual.status_code == 200
    assert manual.json()["title"] == "Manual moment"


def test_failed_job_can_be_retried_without_losing_its_input(
    tmp_path: Path, monkeypatch
) -> None:
    _, database_path = _configure_test_paths(monkeypatch, tmp_path)
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"video")
    job_id = "c" * 32
    create_job(job_id, "video.mp4", str(input_path), database_path)
    update_job(
        job_id,
        status="failed",
        progress=70,
        error="Temporary provider error",
        path=database_path,
    )
    scheduled = []
    monkeypatch.setattr(routes, "_schedule_job", lambda *args: scheduled.append(args))

    with TestClient(app, client=("127.0.0.1", 12345)) as client:
        response = client.post(f"/api/jobs/{job_id}/retry")

    assert response.status_code == 202
    assert response.json()["status"] == "queued"
    assert response.json()["progress"] == 0
    assert response.json()["error"] is None
    assert routes.get_job(job_id, database_path)["input_path"] == str(input_path)
    assert scheduled == [(job_id, "process")]


def test_job_source_video_can_be_saved_by_the_user(
    tmp_path: Path, monkeypatch
) -> None:
    _, database_path = _configure_test_paths(monkeypatch, tmp_path)
    input_path = tmp_path / "source.mp4"
    input_path.write_bytes(b"source video")
    job_id = "e" * 32
    create_job(job_id, "https://youtu.be/example", "", database_path)
    update_job(
        job_id,
        status="ready",
        input_path=str(input_path),
        path=database_path,
    )

    with TestClient(app, client=("127.0.0.1", 12345)) as client:
        job = client.get(f"/api/jobs/{job_id}")
        response = client.get(f"/api/jobs/{job_id}/source-download")

    assert job.status_code == 200
    assert job.json()["source_video_available"] is True
    assert job.json()["source_video_name"] == "source.mp4"
    assert response.status_code == 200
    assert response.content == b"source video"
    assert response.headers["content-disposition"].endswith('filename="source.mp4"')


def test_job_source_download_reports_missing_video(
    tmp_path: Path, monkeypatch
) -> None:
    _, database_path = _configure_test_paths(monkeypatch, tmp_path)
    job_id = "f" * 32
    create_job(job_id, "https://youtu.be/example", "", database_path)

    with TestClient(app, client=("127.0.0.1", 12345)) as client:
        response = client.get(f"/api/jobs/{job_id}/source-download")

    assert response.status_code == 404
    assert "No se encontró el vídeo original" in response.json()["detail"]


def test_history_cleanup_removes_job_files_but_preserves_exports(
    tmp_path: Path, monkeypatch
) -> None:
    _, database_path = _configure_test_paths(monkeypatch, tmp_path)
    output = tmp_path / "output"
    job_id = "d" * 32
    job_dir = output / "jobs" / job_id
    job_dir.mkdir(parents=True)
    input_path = job_dir / "input.mp4"
    input_path.write_bytes(b"source video")
    create_job(job_id, "video.mp4", str(input_path), database_path)
    update_job(job_id, status="ready", path=database_path)
    save_results(
        job_id,
        [Clip(0, 35, "A moment", "A reason.", 8)],
        [TranscriptWord(0, 1, "Hello.")],
        database_path,
    )
    exported_video = output / "clips" / "exported.mp4"
    exported_thumbnail = output / "thumbnails" / "exported.png"
    exported_video.parent.mkdir(parents=True)
    exported_thumbnail.parent.mkdir(parents=True)
    exported_video.write_bytes(b"rendered")
    exported_thumbnail.write_bytes(b"thumbnail")
    save_clip_export(
        job_id, 1, exported_video, exported_thumbnail, database_path
    )

    with TestClient(app, client=("127.0.0.1", 12345)) as client:
        response = client.delete("/api/jobs")
        jobs = client.get("/api/jobs")

    assert response.status_code == 200
    assert response.json() == {"deleted": 1}
    assert jobs.json() == []
    assert not job_dir.exists()
    assert exported_video.read_bytes() == b"rendered"
    assert exported_thumbnail.read_bytes() == b"thumbnail"


def test_history_cleanup_rejects_active_jobs_without_deleting_files(
    tmp_path: Path, monkeypatch
) -> None:
    _, database_path = _configure_test_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(routes, "_schedule_job", lambda *_args: None)
    job_id = "e" * 32
    job_dir = tmp_path / "output" / "jobs" / job_id
    job_dir.mkdir(parents=True)
    marker = job_dir / "keep.txt"
    marker.write_text("active", encoding="utf-8")
    create_job(job_id, "video.mp4", "", database_path)
    update_job(job_id, status="queued", path=database_path)

    with TestClient(app, client=("127.0.0.1", 12345)) as client:
        response = client.delete("/api/jobs")

    assert response.status_code == 409
    assert marker.read_text(encoding="utf-8") == "active"
    assert routes.get_job(job_id, database_path)["status"] == "queued"


def test_video_upload_is_saved_under_a_generated_job_directory(
    tmp_path: Path, monkeypatch
) -> None:
    _, database_path = _configure_test_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(routes, "DEFAULT_DATABASE_PATH", database_path)
    monkeypatch.setattr(routes, "_schedule_job", lambda *args: None)

    with TestClient(app, client=("127.0.0.1", 12345)) as client:
        response = client.post(
            "/api/jobs",
            files={"video": ("sample.mp4", b"sample video bytes", "video/mp4")},
        )

    assert response.status_code == 202
    job_id = response.json()["id"]
    stored_path = tmp_path / "output" / "jobs" / job_id / "input.mp4"
    assert stored_path.read_bytes() == b"sample video bytes"


def test_youtube_shorthand_is_accepted_and_stored_as_https(
    tmp_path: Path, monkeypatch
) -> None:
    _, database_path = _configure_test_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(routes, "DEFAULT_DATABASE_PATH", database_path)
    monkeypatch.setattr(routes, "_schedule_job", lambda *args: None)

    with TestClient(app, client=("127.0.0.1", 12345)) as client:
        response = client.post(
            "/api/jobs",
            data={"source": "youtu.be/abc123?t=5"},
        )

    assert response.status_code == 202
    job = routes.get_job(response.json()["id"], database_path)
    assert job["source"] == "https://youtu.be/abc123?t=5"


def test_clip_source_preview_is_available_after_analysis(
    tmp_path: Path, monkeypatch
) -> None:
    _, database_path = _configure_test_paths(monkeypatch, tmp_path)
    video_path = tmp_path / "source.mp4"
    video_path.write_bytes(b"sample video bytes")
    job_id = "b" * 32
    create_job(job_id, "source.mp4", str(video_path), database_path)
    update_job(job_id, status="ready", path=database_path)
    words = [
        TranscriptWord(0, 1, "Opening."),
        TranscriptWord(5, 6, "A"),
        TranscriptWord(10, 11, "moment."),
        TranscriptWord(35, 36, "Closing."),
    ]
    save_results(
        job_id,
        [Clip(5, 35, "Suggested moment", "A reason.", 8)],
        words,
        database_path,
    )
    clip = routes.get_job_clips(job_id, database_path)[0]

    with TestClient(app, client=("127.0.0.1", 12345)) as client:
        response = client.get(
            f"/api/jobs/{job_id}/clips/{clip['id']}/source-preview"
        )
        partial = client.get(
            f"/api/jobs/{job_id}/clips/{clip['id']}/source-preview",
            headers={"Range": "bytes=7-12"},
        )

    assert response.status_code == 200
    assert response.headers["content-type"] == "video/mp4"
    assert response.content == b"sample video bytes"
    assert partial.status_code == 206
    assert partial.content == b"video "


def test_repeated_exports_keep_unique_files_and_serve_the_latest(
    tmp_path: Path, monkeypatch
) -> None:
    _, database_path = _configure_test_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(routes, "DEFAULT_DATABASE_PATH", database_path)
    video_path = tmp_path / "source.mp4"
    video_path.write_bytes(b"source")
    job_id = "c" * 32
    create_job(job_id, "source.mp4", str(video_path), database_path)
    words = [
        TranscriptWord(0, 1, "Opening."),
        TranscriptWord(5, 6, "A"),
        TranscriptWord(10, 11, "moment."),
        TranscriptWord(35, 36, "Closing."),
    ]
    save_results(
        job_id,
        [Clip(5, 35, "Suggested moment", "A reason.", 8)],
        words,
        database_path,
    )
    update_job(job_id, status="ready", path=database_path)
    output_folder = tmp_path / "output"
    config = SimpleNamespace(
        output_folder=output_folder,
        subtitle_style="classic",
        reframe_mode="center",
    )
    render_count = 0
    observed_progress = []

    class Analyzer:
        def close(self):
            pass

    def create_thumbnail(*args):
        thumbnail_path = args[4]
        thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
        thumbnail_path.write_bytes(b"thumbnail")

    def render_clip(*args, **kwargs):
        nonlocal render_count
        render_count += 1
        output_path = args[3]
        progress_callback = kwargs["progress_callback"]
        progress_callback(0.5)
        observed_progress.append(get_job(job_id, database_path)["progress"])
        progress_callback(0.9)
        observed_progress.append(get_job(job_id, database_path)["progress"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(f"export-{render_count}".encode("ascii"))

    monkeypatch.setattr(routes, "_job_config", lambda: config)
    monkeypatch.setattr(routes, "create_analyzer", lambda _config: Analyzer())
    monkeypatch.setattr(routes, "create_thumbnail", create_thumbnail)
    monkeypatch.setattr(routes, "render_clip", render_clip)

    routes._render_job(job_id)
    assert observed_progress[:2] == sorted(observed_progress[:2])
    assert observed_progress[0] > 5
    assert observed_progress[1] > observed_progress[0]
    clip = routes.get_job_clips(job_id, database_path)[0]
    first_export = routes.get_clip_export(job_id, clip["id"], database_path)
    assert first_export is not None
    first_video_path = Path(first_export["video_path"])
    first_thumbnail_path = Path(first_export["thumbnail_path"])
    first_bytes = first_video_path.read_bytes()

    routes._render_job(job_id)
    latest_export = routes.get_clip_export(job_id, clip["id"], database_path)
    assert latest_export is not None
    latest_video_path = Path(latest_export["video_path"])
    latest_thumbnail_path = Path(latest_export["thumbnail_path"])
    assert first_video_path != latest_video_path
    assert first_thumbnail_path != latest_thumbnail_path
    assert first_video_path.read_bytes() == first_bytes
    assert latest_video_path.read_bytes() == b"export-2"

    with TestClient(app, client=("127.0.0.1", 12345)) as client:
        preview = client.get(
            f"/api/jobs/{job_id}/clips/{clip['id']}/preview"
        )
        first_download = client.get(f"/api/jobs/{job_id}/download")
        second_download = client.get(f"/api/jobs/{job_id}/download")

    assert preview.status_code == 200
    assert preview.content == b"export-2"
    assert first_download.status_code == second_download.status_code == 200
    assert first_download.headers["content-disposition"] != second_download.headers["content-disposition"]
    with zipfile.ZipFile(io.BytesIO(first_download.content)) as archive:
        assert latest_video_path.name in archive.namelist()
        assert latest_thumbnail_path.name in archive.namelist()


def test_launcher_creates_a_keyless_user_config_on_first_run(
    tmp_path: Path, monkeypatch
) -> None:
    config_path = tmp_path / "AiClip" / "config.local.json"
    output_folder = tmp_path / "AiClip" / "output"
    monkeypatch.setattr(app_launcher, "APP_DATA_DIR", config_path.parent)
    monkeypatch.setattr(app_launcher, "DEFAULT_CONFIG_PATH", config_path)
    monkeypatch.setattr(app_launcher, "DEFAULT_OUTPUT_FOLDER", output_folder)
    monkeypatch.setattr(app_launcher, "PROJECT_ROOT", Path(__file__).resolve().parents[1])

    app_launcher._initialize_config()

    settings = json.loads(config_path.read_text(encoding="utf-8"))
    assert settings["groq_api_key"] == ""
    assert "gemini_api_key" not in settings
    assert settings["output_folder"] == str(output_folder)
