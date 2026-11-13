import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2


OUTPUT_WIDTH = 1080
OUTPUT_HEIGHT = 1920
FACE_SAMPLE_SECONDS = 0.5
FACE_SMOOTHING = 0.35
MIN_FACE_DETECTIONS = 2
MULTIPLE_FACES_SAMPLE_RATIO = 0.25


class ReframeError(RuntimeError):
    """Error legible al analizar o encuadrar un vídeo vertical."""


@dataclass(frozen=True)
class FaceCenter:
    time: float
    x: float
    y: float


@dataclass(frozen=True)
class ReframePlan:
    mode: Literal["center", "face", "blur"]
    crop_width: int
    crop_height: int
    centers: tuple[FaceCenter, ...] = ()
    fallback_reason: str | None = None


def calculate_crop_size(
    source_width: int,
    source_height: int,
    output_width: int = OUTPUT_WIDTH,
    output_height: int = OUTPUT_HEIGHT,
) -> tuple[int, int]:
    if min(source_width, source_height, output_width, output_height) <= 0:
        raise ValueError("El ancho y el alto deben ser mayores que cero.")
    source_ratio = source_width / source_height
    output_ratio = output_width / output_height
    if source_ratio > output_ratio:
        crop_height = source_height - source_height % 2
        crop_width = int(crop_height * output_ratio) // 2 * 2
    else:
        crop_width = source_width - source_width % 2
        crop_height = int(crop_width / output_ratio) // 2 * 2
    return max(2, crop_width), max(2, crop_height)


def smooth_face_centers(
    detections: list[FaceCenter], alpha: float = FACE_SMOOTHING
) -> tuple[FaceCenter, ...]:
    if not 0 < alpha <= 1:
        raise ValueError("El suavizado debe estar entre cero y uno.")
    if not detections:
        return ()
    ordered = sorted(detections, key=lambda item: item.time)
    result = [ordered[0]]
    for detection in ordered[1:]:
        previous = result[-1]
        result.append(
            FaceCenter(
                time=detection.time,
                x=previous.x + alpha * (detection.x - previous.x),
                y=previous.y + alpha * (detection.y - previous.y),
            )
        )
    return tuple(result)


def build_reframe_plan(
    source_width: int,
    source_height: int,
    mode: Literal["center", "face", "blur"],
    detections: list[FaceCenter] | None = None,
    multiple_people: bool = False,
) -> ReframePlan:
    crop_width, crop_height = calculate_crop_size(source_width, source_height)
    if mode == "center":
        return ReframePlan("center", crop_width, crop_height)
    if mode == "blur":
        return ReframePlan("blur", crop_width, crop_height)
    if mode != "face":
        raise ValueError(f"Modo de encuadre no válido: {mode}")
    if multiple_people:
        return ReframePlan(
            "blur",
            crop_width,
            crop_height,
            fallback_reason="Se detectaron varias personas; se usará fondo desenfocado.",
        )
    if not detections:
        return ReframePlan(
            "blur",
            crop_width,
            crop_height,
            fallback_reason="No se detectó una cara estable; se usará fondo desenfocado.",
        )
    return ReframePlan(
        "face",
        crop_width,
        crop_height,
        smooth_face_centers(detections),
    )


def analyze_face_tracking(
    video_path: Path, clip_start: float, clip_duration: float
) -> ReframePlan:
    if clip_start < 0 or clip_duration <= 0:
        raise ValueError("El inicio y la duración del clip no son válidos.")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ReframeError("No se pudo abrir el vídeo para detectar caras.")
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if width <= 0 or height <= 0 or not math.isfinite(fps) or fps <= 0:
            raise ReframeError("No se pudieron leer las dimensiones del vídeo.")
        cascade_dir = Path(cv2.data.haarcascades)
        frontal_detector = cv2.CascadeClassifier(
            str(cascade_dir / "haarcascade_frontalface_default.xml")
        )
        if frontal_detector.empty():
            raise ReframeError(
                "OpenCV no pudo cargar el detector de rostros integrado."
            )
        alternate_detector = cv2.CascadeClassifier(
            str(cascade_dir / "haarcascade_frontalface_alt2.xml")
        )
        if alternate_detector.empty():
            alternate_detector = None
        profile_detector = cv2.CascadeClassifier(
            str(cascade_dir / "haarcascade_profileface.xml")
        )
        if profile_detector.empty():
            profile_detector = None

        sample_count = max(1, math.ceil(clip_duration / FACE_SAMPLE_SECONDS))
        detections: list[FaceCenter] = []
        multiple_face_samples = 0
        multiple_face_threshold = max(
            2,
            math.ceil(sample_count * MULTIPLE_FACES_SAMPLE_RATIO),
        )
        scale = min(1.0, 800 / max(width, height))
        for index in range(sample_count):
            time_in_clip = min(index * FACE_SAMPLE_SECONDS, clip_duration)
            frame_index = round((clip_start + time_in_clip) * fps)
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = capture.read()
            if not success:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if scale < 1:
                gray = cv2.resize(
                    gray,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_AREA,
                )
            gray = cv2.equalizeHist(gray)
            faces = _detect_faces(
                gray,
                frontal_detector,
                alternate_detector,
                profile_detector,
            )
            if len(faces) > 1:
                multiple_face_samples += 1
                if multiple_face_samples >= multiple_face_threshold:
                    break
            if faces:
                x, y, face_width, face_height = max(
                    faces,
                    key=lambda face: face[2] * face[3],
                )
                detections.append(
                    FaceCenter(
                        time=time_in_clip,
                        x=(x + face_width / 2) / scale,
                        y=(y + face_height / 2) / scale,
                    )
                )
        if len(detections) < MIN_FACE_DETECTIONS:
            detections = []
        return build_reframe_plan(
            width,
            height,
            "face",
            detections=detections,
            multiple_people=multiple_face_samples >= multiple_face_threshold,
        )
    except cv2.error as exc:
        raise ReframeError("OpenCV no pudo analizar las caras del vídeo.") from exc
    finally:
        capture.release()


def _detect_faces(
    gray,
    frontal_detector,
    alternate_detector,
    profile_detector,
) -> list[tuple[int, int, int, int]]:
    _, width = gray.shape
    inputs = [("frontal", frontal_detector, gray, False)]
    if alternate_detector is not None:
        inputs.append(("alternate", alternate_detector, gray, False))
    if profile_detector is not None:
        inputs.extend(
            [
                ("profile", profile_detector, gray, False),
                ("profile", profile_detector, cv2.flip(gray, 1), True),
            ]
        )

    detections: list[tuple[tuple[int, int, int, int], set[str]]] = []
    for detector_name, detector, image, mirrored in inputs:
        for x, y, face_width, face_height in detector.detectMultiScale(
            image,
            scaleFactor=1.08,
            minNeighbors=4,
            minSize=(24, 24),
        ):
            if mirrored:
                x = width - x - face_width
            candidate = (int(x), int(y), int(face_width), int(face_height))
            match = next(
                (
                    voters
                    for face, voters in detections
                    if _face_overlap(candidate, face) >= 0.6
                ),
                None,
            )
            if match is None:
                detections.append((candidate, {detector_name}))
            else:
                match.add(detector_name)

    confirmed = [face for face, voters in detections if len(voters) >= 2]
    if confirmed:
        return confirmed
    if detections:
        largest = max(
            detections,
            key=lambda detection: detection[0][2] * detection[0][3],
        )
        return [largest[0]]
    return []


def _face_overlap(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    first_x, first_y, first_width, first_height = first
    second_x, second_y, second_width, second_height = second
    intersection_width = max(
        0,
        min(first_x + first_width, second_x + second_width)
        - max(first_x, second_x),
    )
    intersection_height = max(
        0,
        min(first_y + first_height, second_y + second_height)
        - max(first_y, second_y),
    )
    intersection = intersection_width * intersection_height
    smaller_face_area = min(
        first_width * first_height,
        second_width * second_height,
    )
    return intersection / smaller_face_area if smaller_face_area else 0


def build_crop_filter(plan: ReframePlan) -> str:
    if plan.mode == "blur":
        raise ValueError("El modo de fondo desenfocado no usa un recorte simple.")
    if plan.mode == "center" or not plan.centers:
        x_expression = f"(iw-{plan.crop_width})/2"
        y_expression = f"(ih-{plan.crop_height})/2"
    else:
        x_expression = _position_expression(
            plan.centers, "x", "iw", plan.crop_width
        )
        y_expression = _position_expression(
            plan.centers, "y", "ih", plan.crop_height
        )
    return (
        f"crop={plan.crop_width}:{plan.crop_height}:"
        f"{x_expression}:{y_expression},"
        f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:flags=lanczos"
    )


def _position_expression(
    centers: tuple[FaceCenter, ...],
    axis: Literal["x", "y"],
    source_size: Literal["iw", "ih"],
    crop_size: int,
) -> str:
    crop_half = crop_size / 2
    value = lambda center: (
        f"max(0\\,min({source_size}-{crop_size}\\,"
        f"{getattr(center, axis) - crop_half:.3f}))"
    )
    if len(centers) == 1:
        return value(centers[0])

    expression = value(centers[-1])
    for left, right in reversed(list(zip(centers, centers[1:]))):
        duration = right.time - left.time
        if duration <= 0:
            continue
        start_value = value(left)
        end_value = value(right)
        interpolation = (
            f"({start_value}+({end_value}-{start_value})"
            f"*(t-{left.time:.3f})/{duration:.3f})"
        )
        expression = f"if(lt(t\\,{right.time:.3f})\\,{interpolation}\\,{expression})"
    first = value(centers[0])
    return f"if(lt(t\\,{centers[0].time:.3f})\\,{first}\\,{expression})"


def probe_nvenc(ffmpeg: str) -> bool:
    try:
        result = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=64x64:d=0.1",
                "-frames:v",
                "1",
                "-c:v",
                "h264_nvenc",
                "-f",
                "null",
                "-",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0
