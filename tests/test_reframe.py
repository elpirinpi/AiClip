import pytest

from app.pipeline.reframe import (
    FaceCenter,
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
