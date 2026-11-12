import json
from pathlib import Path

from fastapi.testclient import TestClient

import app.api.routes as routes
import app.config as config_module
import scripts.start_app as app_launcher
from app.db import create_job, save_results, update_job
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
    assert "El audio se envía a Groq" in home.text
    assert "eres responsable de tener derecho a procesar el contenido" in home.text
    assert health.json() == {"status": "ok"}


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
    assert settings["gemini_api_key"] == ""
    assert settings["output_folder"] == str(output_folder)
