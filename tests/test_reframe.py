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


def test_face_tracking_uses_profile_faces_and_samples_in_clip_time(
    monkeypatch,
) -> None:
    import cv2
    import numpy as np

    class Capture:
        frame_index = 0
        released = False

        def isOpened(self):
            return True

        def get(self, prop):
            return {
                cv2.CAP_PROP_FRAME_WIDTH: 1920,
                cv2.CAP_PROP_FRAME_HEIGHT: 1080,
                cv2.CAP_PROP_FPS: 2,
            }[prop]

        def set(self, _prop, frame_index):
            self.frame_index = int(frame_index)
            return True

        def read(self):
            return True, np.zeros((1080, 1920, 3), dtype=np.uint8)

        def release(self):
            self.released = True

    capture = Capture()

    class Detector:
        def __init__(self, path):
            self.name = (
                "profile"
                if "profileface" in path
                else "alternate"
                if "alt2" in path
                else "frontal"
            )
            self.calls_by_frame = {}

        def empty(self):
            return False

        def detectMultiScale(self, _image, **_options):
            if self.name == "frontal":
                return ()
            x = 40 + (capture.frame_index - 4) * 10
            if self.name == "alternate":
                return np.array([[x, 20, 40, 40]])
            calls = self.calls_by_frame.get(capture.frame_index, 0)
            self.calls_by_frame[capture.frame_index] = calls + 1
            if calls % 2:
                return ()
            return np.array([[x, 20, 40, 40]])

    monkeypatch.setattr(reframe.cv2, "VideoCapture", lambda _path: capture)
    monkeypatch.setattr(reframe.cv2, "CascadeClassifier", Detector)

    plan = analyze_face_tracking(Path("video.mp4"), 2, 1.5)

    assert plan.mode == "face"
    assert [center.time for center in plan.centers] == [0, 0.5, 1.0]
    assert plan.centers[0].x == pytest.approx(144)
    assert plan.centers[1].x > plan.centers[0].x
    assert capture.released


def test_one_transient_multiple_face_detection_does_not_disable_tracking(
    monkeypatch,
) -> None:
    import cv2
    import numpy as np

    class Capture:
        frame_index = 0

        def isOpened(self):
            return True

        def get(self, prop):
            return {
                cv2.CAP_PROP_FRAME_WIDTH: 320,
                cv2.CAP_PROP_FRAME_HEIGHT: 180,
                cv2.CAP_PROP_FPS: 2,
            }[prop]

        def set(self, _prop, frame_index):
            self.frame_index = int(frame_index)
            return True

        def read(self):
            return True, np.zeros((180, 320, 3), dtype=np.uint8)

        def release(self):
            pass

    capture = Capture()

    class Detector:
        def __init__(self, path):
            self.name = (
                "profile"
                if "profileface" in path
                else "alternate"
                if "alt2" in path
                else "frontal"
            )
            self.calls_by_frame = {}

        def empty(self):
            return False

        def detectMultiScale(self, _image, **_options):
            if self.name == "frontal":
                return ()
            if capture.frame_index == 0:
                faces = np.array([[10, 10, 30, 30], [100, 10, 30, 30]])
            else:
                faces = np.array([[10 + capture.frame_index, 10, 30, 30]])
            if self.name == "alternate":
                return faces
            calls = self.calls_by_frame.get(capture.frame_index, 0)
            self.calls_by_frame[capture.frame_index] = calls + 1
            if calls % 2:
                return ()
            return faces

    monkeypatch.setattr(reframe.cv2, "VideoCapture", lambda _path: capture)
    monkeypatch.setattr(reframe.cv2, "CascadeClassifier", Detector)

    plan = analyze_face_tracking(Path("video.mp4"), 0, 3)

    assert plan.mode == "face"
    assert len(plan.centers) == 6


def test_persistent_multiple_face_detections_use_blurred_background(
    monkeypatch,
) -> None:
    import cv2
    import numpy as np

    class Capture:
        frame_index = 0

        def isOpened(self):
            return True

        def get(self, prop):
            return {
                cv2.CAP_PROP_FRAME_WIDTH: 320,
                cv2.CAP_PROP_FRAME_HEIGHT: 180,
                cv2.CAP_PROP_FPS: 2,
            }[prop]

        def set(self, _prop, frame_index):
            self.frame_index = int(frame_index)
            return True

        def read(self):
            return True, np.zeros((180, 320, 3), dtype=np.uint8)

        def release(self):
            pass

    capture = Capture()

    class Detector:
        def __init__(self, path):
            self.name = (
                "profile"
                if "profileface" in path
                else "alternate"
                if "alt2" in path
                else "frontal"
            )
            self.calls_by_frame = {}

        def empty(self):
            return False

        def detectMultiScale(self, _image, **_options):
            if self.name == "frontal":
                return ()
            faces = np.array([[10, 10, 30, 30], [100, 10, 30, 30]])
            if self.name == "alternate":
                return faces
            calls = self.calls_by_frame.get(capture.frame_index, 0)
            self.calls_by_frame[capture.frame_index] = calls + 1
            if calls % 2:
                return ()
            return faces

    monkeypatch.setattr(reframe.cv2, "VideoCapture", lambda _path: capture)
    monkeypatch.setattr(reframe.cv2, "CascadeClassifier", Detector)

    plan = analyze_face_tracking(Path("video.mp4"), 0, 3)

    assert plan.mode == "blur"
    assert "varias personas" in plan.fallback_reason


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
