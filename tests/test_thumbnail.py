import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from app.pipeline.segment import Clip
from app.pipeline.thumbnail import (
    FrameCandidate,
    ThumbnailError,
    _score_frame,
    compose_thumbnail,
    generate_thumbnail_text,
)
from app.providers.base import TranscriptWord


class StaticProvider:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompt = ""

    def generate(self, prompt: str) -> str:
        self.prompt = prompt
        return self.response


def test_thumbnail_title_is_limited_to_five_words_and_uses_clip_transcript() -> None:
    provider = StaticProvider(
        json.dumps({"text": "  Escucha estas grandes ideas sorprendentes ahora  "})
    )
    clip = Clip(10, 20, "Title", "Reason.", 8)
    words = [
        TranscriptWord(1, 2, "Antes."),
        TranscriptWord(10, 11, "Escucha"),
        TranscriptWord(12, 13, "estas"),
        TranscriptWord(19, 20, "ideas."),
        TranscriptWord(21, 22, "Después."),
    ]

    title = generate_thumbnail_text(clip, words, provider)

    assert title == "Escucha estas grandes ideas sorprendentes"
    assert "Antes." not in provider.prompt
    assert "Después." not in provider.prompt
    assert "Escucha estas ideas." in provider.prompt


@pytest.mark.parametrize("response", ["not-json", "{}", '{"text": ""}'])
def test_thumbnail_title_rejects_malformed_provider_responses(response: str) -> None:
    with pytest.raises(ThumbnailError):
        generate_thumbnail_text(
            Clip(0, 10, "Title", "Reason.", 8),
            [TranscriptWord(0, 1, "Hello.")],
            StaticProvider(response),
        )


def test_frame_score_prefers_visible_face_and_two_detected_eyes() -> None:
    class FakeDetector:
        def __init__(self, detections: list[tuple[int, int, int, int]]) -> None:
            self.detections = np.array(detections, dtype=np.int32)

        def detectMultiScale(self, image, **kwargs):
            return self.detections

    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    face = _score_frame(
        frame,
        0,
        1,
        FakeDetector([(200, 80, 120, 120)]),
        FakeDetector([(20, 20, 30, 20), (70, 20, 30, 20)]),
    )
    no_face = _score_frame(
        frame,
        1,
        1,
        FakeDetector([]),
        FakeDetector([]),
    )

    assert face.face is not None
    assert face.eyes_visible
    assert face.score > no_face.score
    assert face.frame.shape[:2] == (360, 202)


def test_compose_thumbnail_writes_readable_vertical_png(tmp_path: Path) -> None:
    frame = FrameCandidate(
        time=1.5,
        score=10,
        frame=np.full((360, 202, 3), (30, 90, 180), dtype=np.uint8),
        face=(100, 150, 70, 70),
        eyes_visible=True,
    )

    output = compose_thumbnail(frame, "Una pausa transforma todo", tmp_path / "thumb.png")

    with Image.open(output) as image:
        assert image.size == (1080, 1920)
        assert image.format == "PNG"
        assert np.asarray(image).std() > 0
