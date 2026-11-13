from pathlib import Path
from io import StringIO

from app.config import AppConfig
from app.pipeline.reframe import FaceCenter, ReframePlan
from app.pipeline.render import (
    _run_ffmpeg_with_progress,
    _select_video_encoder,
    _video_filter,
    render_clip,
)
from app.pipeline.segment import Clip


def test_blurred_background_maps_only_processed_video() -> None:
    option, graph, video_map = _video_filter(
        ReframePlan("blur", 202, 360),
        "captions.ass",
    )

    assert option == "-filter_complex"
    assert "[bg]scale=135:240:force_original_aspect_ratio=increase," in graph
    assert "crop=135:240,boxblur=4:2,scale=1080:1920[blurred]" in graph
    assert "[blurred][fit]overlay=(W-w)/2:(H-h)/2" in graph
    assert video_map == "[outv]"


def test_face_filter_uses_smoothed_crop_expression() -> None:
    option, video_filter, video_map = _video_filter(
        ReframePlan(
            "face",
            202,
            360,
            (FaceCenter(0, 320, 150), FaceCenter(1, 420, 160)),
        ),
        "captions.ass",
    )

    assert option == "-vf"
    assert "crop=202:360:" in video_filter
    assert "if(lt(t\\,1.000)" in video_filter
    assert video_map is None


def test_render_passes_long_filter_graph_through_script_file(
    monkeypatch, tmp_path: Path
) -> None:
    from app.pipeline import render

    video_path = tmp_path / "input.mp4"
    video_path.touch()
    output_path = tmp_path / "output" / "clip.mp4"
    work_dir = tmp_path / "work"
    config = AppConfig(
        groq_api_key="",
        gemini_api_key="",
        ffmpeg_path="ffmpeg",
        output_folder=tmp_path / "output",
    )
    long_filter = "scale=1080:1920," + "hflip," * 5000 + "null"
    captured_command: list[str] = []
    captured_script = ""

    monkeypatch.setattr(render, "_find_executable", lambda _value: "ffmpeg.exe")
    monkeypatch.setattr(render, "write_ass_subtitles", lambda *_args: None)
    monkeypatch.setattr(render, "_video_dimensions", lambda _path: (1920, 1080))
    monkeypatch.setattr(
        render, "_video_filter", lambda *_args: ("-vf", long_filter, None)
    )
    monkeypatch.setattr(render, "_select_video_encoder", lambda _ffmpeg: "libx264")

    def run_with_progress(command, _work_dir, _duration, progress_callback):
        nonlocal captured_script
        captured_command.extend(command)
        script_path = work_dir / command[command.index("-filter_complex_script") + 1]
        captured_script = script_path.read_text(encoding="utf-8")
        if progress_callback is not None:
            progress_callback(0.5)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"rendered")

    monkeypatch.setattr(render, "_run_ffmpeg_with_progress", run_with_progress)

    result = render_clip(
        video_path,
        Clip(0, 30, "Prueba", "Filtro extenso", 10),
        [],
        output_path,
        work_dir,
        "classic",
        config,
    )

    assert result == output_path
    assert "-filter_complex_script" in captured_command
    assert captured_command[
        captured_command.index("-map") + 1
    ] == "[outv]"
    assert "[0:v]scale=1080:1920," in captured_script
    assert captured_script.endswith("[outv]\n")
    assert len(" ".join(captured_command)) < 2000


def test_ffmpeg_progress_output_reports_render_fraction(monkeypatch, tmp_path: Path) -> None:
    from app.pipeline import render

    class Process:
        stdout = StringIO(
            "out_time_us=500000\nprogress=continue\n"
            "out_time_us=1500000\nprogress=end\n"
        )

        def poll(self):
            return 0

        def wait(self):
            return 0

    monkeypatch.setattr(render.subprocess, "Popen", lambda *_args, **_kwargs: Process())
    progress = []

    _run_ffmpeg_with_progress(["ffmpeg"], tmp_path, 2, progress.append)

    assert progress == [0.25, 0.75]


def test_encoder_selection_prefers_nvenc_only_when_probe_succeeds(monkeypatch) -> None:
    from app.pipeline import render

    _select_video_encoder.cache_clear()
    monkeypatch.setattr(render, "probe_nvenc", lambda ffmpeg: True)
    assert _select_video_encoder("ffmpeg-test") == "h264_nvenc"

    _select_video_encoder.cache_clear()
    monkeypatch.setattr(render, "probe_nvenc", lambda ffmpeg: False)
    assert _select_video_encoder("ffmpeg-test") == "libx264"
    _select_video_encoder.cache_clear()
