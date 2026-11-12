from pathlib import Path

from app.pipeline.reframe import FaceCenter, ReframePlan
from app.pipeline.render import _select_video_encoder, _video_filter


def test_blurred_background_maps_only_processed_video() -> None:
    option, graph, video_map = _video_filter(
        ReframePlan("blur", 202, 360),
        "captions.ass",
    )

    assert option == "-filter_complex"
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


def test_encoder_selection_prefers_nvenc_only_when_probe_succeeds(monkeypatch) -> None:
    from app.pipeline import render

    _select_video_encoder.cache_clear()
    monkeypatch.setattr(render, "probe_nvenc", lambda ffmpeg: True)
    assert _select_video_encoder("ffmpeg-test") == "h264_nvenc"

    _select_video_encoder.cache_clear()
    monkeypatch.setattr(render, "probe_nvenc", lambda ffmpeg: False)
    assert _select_video_encoder("ffmpeg-test") == "libx264"
    _select_video_encoder.cache_clear()
