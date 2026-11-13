from pathlib import Path

import pytest

import app.pipeline.reframe as reframe
from app.pipeline.reframe import (
    FaceCenter,
    analyze_face_tracking,
    build_crop_filter,
    build_reframe_plan,
    calculate_crop_size,
    smooth_face_centers,
)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ((1920, 1080), (606, 1080)),
        ((1080, 1920), (1080, 1920)),
        ((1080, 1080), (606, 1080)),
        ((1080, 1350), (758, 1350)),
        ((720, 1280), (720, 1280)),
    ],
)
def test_vertical_crop_preserves_source_and_output_aspect_ratio(
    source: tuple[int, int], expected: tuple[int, int]
) -> None:
    crop = calculate_crop_size(*source)

    assert crop == expected
    assert crop[0] % 2 == crop[1] % 2 == 0
    assert crop[0] <= source[0] and crop[1] <= source[1]
    assert crop[0] / crop[1] == pytest.approx(9 / 16, abs=0.002)


def test_center_mode_uses_fixed_center_crop() -> None:
    plan = build_reframe_plan(1920, 1080, "center")

    assert plan.mode == "center"
    assert build_crop_filter(plan) == (
        "crop=606:1080:(iw-606)/2:(ih-1080)/2,"
        "scale=1080:1920:flags=lanczos"
    )


def test_face_mode_smooths_crop_positions() -> None:
    plan = build_reframe_plan(
        1920,
        1080,
        "face",
        detections=[
            FaceCenter(0, 500, 300),
            FaceCenter(1, 900, 500),
        ],
    )
    filter_graph = build_crop_filter(plan)

    assert plan.mode == "face"
    assert plan.centers[1].x == pytest.approx(640)
    assert plan.centers[1].y == pytest.approx(370)
    assert "if(lt(t\\,1.000)" in filter_graph
    assert "crop=606:1080:" in filter_graph


def test_face_position_expression_has_balanced_conditions_for_many_samples() -> None:
    centers = tuple(FaceCenter(index * 0.5, 400 + index, 500) for index in range(120))

    expression = reframe._position_expression(centers, "x", "iw", 606)

    assert expression.count("if(lt(") == 2 * len(centers) - 2
    assert len(expression) < 40_000
    depth = maximum_if_depth(expression)
    assert depth <= 9


def maximum_if_depth(expression: str) -> int:
    depth = 0
    maximum = 0
    parentheses = []
    index = 0
    while index < len(expression):
        if expression.startswith("if(", index):
            parentheses.append(True)
            depth += 1
            maximum = max(maximum, depth)
            index += 3
            continue
        if expression[index] == "(":
            parentheses.append(False)
        elif expression[index] == ")":
            if parentheses.pop():
                depth -= 1
        index += 1
    return maximum


def test_face_mode_uses_blurred_background_without_face_or_with_multiple_people() -> None:
    no_face = build_reframe_plan(1920, 1080, "face")
    group = build_reframe_plan(
        1920, 1080, "face", [FaceCenter(0, 400, 400)], multiple_people=True
    )

    assert no_face.mode == "blur"
    assert "No se detectó una cara" in no_face.fallback_reason
    assert group.mode == "blur"
    assert "varias personas" in group.fallback_reason


def test_smoothing_rejects_out_of_range_alpha() -> None:
    with pytest.raises(ValueError, match="suavizado"):
        smooth_face_centers([FaceCenter(0, 10, 10)], alpha=0)


def _mock_face_sample_extraction(
    monkeypatch,
    source_width: int,
    source_height: int,
    frame_count: int,
):
    import cv2
    import numpy as np
    from types import SimpleNamespace

    class Capture:
        released = 0

        def isOpened(self):
            return True

        def get(self, prop):
            return {
                cv2.CAP_PROP_FRAME_WIDTH: source_width,
                cv2.CAP_PROP_FRAME_HEIGHT: source_height,
            }[prop]

        def release(self):
            self.released += 1

    capture = Capture()
    scaled_width = min(reframe.FACE_SAMPLE_WIDTH, source_width)
    scaled_height = round(source_height * scaled_width / source_width / 2) * 2
    samples = (
        np.stack(
            [
                np.full((scaled_height, scaled_width), index, dtype=np.uint8)
                for index in range(frame_count)
            ]
        )
        if frame_count
        else np.empty((0, scaled_height, scaled_width), dtype=np.uint8)
    )
    commands = []

    def run(command, **_options):
        commands.append(command)
        return SimpleNamespace(stdout=samples.tobytes())

    class Detector:
        def empty(self):
            return False

    monkeypatch.setattr(reframe.cv2, "VideoCapture", lambda _path: capture)
    monkeypatch.setattr(reframe.cv2, "CascadeClassifier", lambda _path: Detector())
    monkeypatch.setattr(reframe.cv2, "equalizeHist", lambda frame: frame)
    monkeypatch.setattr(reframe.subprocess, "run", run)
    monkeypatch.setattr(
        reframe, "_detect_faces", lambda gray, *_detectors: [(20, 20, 40, 40)]
    )
    return capture, commands


def test_face_tracking_extracts_samples_once_and_scales_face_coordinates(
    monkeypatch,
) -> None:
    capture, commands = _mock_face_sample_extraction(monkeypatch, 1920, 1080, 3)
    frames_seen = []

    def detect_faces(gray, *_detectors):
        frames_seen.append(int(gray[0, 0]))
        x = 24 + len(frames_seen) * 4
        return [(x, 20, 40, 40)]

    monkeypatch.setattr(reframe, "_detect_faces", detect_faces)
    plan = analyze_face_tracking(Path("video.mp4"), 2, 1.5)

    assert plan.mode == "face"
    assert [center.time for center in plan.centers] == [0, 0.5, 1.0]
    assert plan.centers[0].x == pytest.approx(144)
    assert plan.centers[1].x > plan.centers[0].x
    assert plan.centers[0].y == pytest.approx(120)
    assert frames_seen == [0, 1, 2]
    assert len(commands) == 1
    assert commands[0][commands[0].index("-vf") + 1] == (
        "setpts=PTS-STARTPTS,fps=2,scale=640:-2"
    )
    assert commands[0][commands[0].index("-ss") + 1] == "2.000"
    assert commands[0][commands[0].index("-t") + 1] == "1.500"
    assert capture.released >= 1


def test_one_transient_multiple_face_detection_does_not_disable_tracking(
    monkeypatch,
) -> None:
    _mock_face_sample_extraction(monkeypatch, 320, 180, 6)

    def detect_faces(gray, *_detectors):
        if int(gray[0, 0]) == 0:
            return [(10, 10, 30, 30), (100, 10, 30, 30)]
        return [(10, 10, 30, 30)]

    monkeypatch.setattr(reframe, "_detect_faces", detect_faces)

    plan = analyze_face_tracking(Path("video.mp4"), 0, 3)

    assert plan.mode == "face"
    assert len(plan.centers) == 6


def test_persistent_multiple_face_detections_use_blurred_background(
    monkeypatch,
) -> None:
    _mock_face_sample_extraction(monkeypatch, 320, 180, 6)
    monkeypatch.setattr(
        reframe,
        "_detect_faces",
        lambda _gray, *_detectors: [(10, 10, 30, 30), (100, 10, 30, 30)],
    )

    plan = analyze_face_tracking(Path("video.mp4"), 0, 3)

    assert plan.mode == "blur"
    assert "varias personas" in plan.fallback_reason


def test_face_tracking_falls_back_when_ffmpeg_returns_no_frames(monkeypatch) -> None:
    _mock_face_sample_extraction(monkeypatch, 320, 180, 0)

    plan = analyze_face_tracking(Path("video.mp4"), 0, 3)

    assert plan.mode == "blur"
    assert "No se detectó una cara" in plan.fallback_reason


def test_face_detector_ignores_unconfirmed_false_secondary_detection() -> None:
    import numpy as np

    class Detector:
        def __init__(self, faces):
            self.faces = faces
            self.calls = 0

        def detectMultiScale(self, _image, **_options):
            self.calls += 1
            if self.calls > 1:
                return ()
            return np.array(self.faces)

    gray = np.zeros((360, 640), dtype=np.uint8)
    frontal = Detector([[412, 279, 60, 60], [341, 75, 122, 122]])
    alternate = Detector([[343, 75, 125, 125]])
    profile = Detector([[315, 64, 139, 139]])

    faces = reframe._detect_faces(gray, frontal, alternate, profile)

    assert len(faces) == 1
    assert faces[0][0] < 360
    assert faces[0][1] < 200
