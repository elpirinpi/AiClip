import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.pipeline.reframe import OUTPUT_HEIGHT, OUTPUT_WIDTH, calculate_crop_size
from app.pipeline.segment import Clip
from app.providers.base import AnalysisProvider, TranscriptWord
from app.providers.base import ProviderError


THUMBNAIL_SAMPLE_LIMIT = 40
THUMBNAIL_SAMPLE_SECONDS = 0.5
THUMBNAIL_TEXT_LIMIT = 5
THUMBNAIL_FONT_SIZE = 88


class ThumbnailError(RuntimeError):
    """Error legible al seleccionar o componer una miniatura."""


@dataclass(frozen=True)
class FrameCandidate:
    time: float
    score: float
    frame: np.ndarray
    face: tuple[int, int, int, int] | None
    eyes_visible: bool


def generate_thumbnail_text(
    clip: Clip,
    words: list[TranscriptWord],
    provider: AnalysisProvider,
) -> str:
    transcript = " ".join(
        word.text
        for word in words
        if word.text.strip() and word.end > clip.start and word.start < clip.end
    )
    if not transcript:
        raise ThumbnailError(
            "El clip no tiene transcripción para proponer el texto de miniatura."
        )
    prompt = (
        "Escribe un texto breve para la miniatura vertical de un clip. "
        "Devuelve únicamente un objeto JSON con la propiedad text. "
        "Usa de 1 a 5 palabras, en español, llamativas pero fieles a la "
        "transcripción. No uses comillas ni afirmaciones que la transcripción "
        "no respalde.\n\n"
        f"Transcripción:\n{transcript}"
    )
    try:
        response = provider.generate(prompt)
    except ProviderError as exc:
        raise ThumbnailError(str(exc)) from exc
    try:
        payload = json.loads(response)
    except (ValueError, TypeError) as exc:
        raise ThumbnailError(
            "Gemini no devolvió JSON válido para el texto de miniatura."
        ) from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
        raise ThumbnailError(
            "Gemini no devolvió el texto de miniatura en el formato esperado."
        )
    title = re.sub(r"\s+", " ", payload["text"]).strip()
    title = title.strip(" \"'“”‘’")
    if not title:
        raise ThumbnailError("Gemini devolvió vacío el texto de miniatura.")
    title = " ".join(title.split()[:THUMBNAIL_TEXT_LIMIT])
    return title


def select_best_frame(
    video_path: Path,
    clip_start: float,
    clip_end: float,
) -> FrameCandidate:
    if not math.isfinite(clip_start) or not math.isfinite(clip_end) or clip_end <= clip_start:
        raise ValueError("El intervalo del clip no es válido para elegir una miniatura.")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ThumbnailError("No se pudo abrir el vídeo para elegir la miniatura.")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0 or total_frames <= 0:
            raise ThumbnailError("No se pudo leer la duración del vídeo para la miniatura.")

        face_detector, eye_detector = _load_detectors()
        duration = clip_end - clip_start
        sample_count = min(
            THUMBNAIL_SAMPLE_LIMIT,
            max(1, math.ceil(duration / THUMBNAIL_SAMPLE_SECONDS)),
        )
        candidates: list[FrameCandidate] = []
        scale = 640 / max(
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        scale = min(1.0, scale)
        for index in range(sample_count):
            time_in_clip = (
                0.0
                if sample_count == 1
                else duration * index / (sample_count - 1)
            )
            frame_index = round((clip_start + time_in_clip) * fps)
            if frame_index >= total_frames:
                continue
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = capture.read()
            if not success:
                continue
            candidates.append(
                _score_frame(
                    frame,
                    time_in_clip,
                    scale,
                    face_detector,
                    eye_detector,
                )
            )
        if not candidates:
            raise ThumbnailError("No se pudo leer ningún fotograma del clip.")
        return max(candidates, key=lambda candidate: candidate.score)
    except cv2.error as exc:
        raise ThumbnailError("OpenCV no pudo analizar los fotogramas del clip.") from exc
    finally:
        capture.release()


def compose_thumbnail(
    frame: FrameCandidate,
    text: str,
    output_path: Path,
) -> Path:
    if not text.strip():
        raise ValueError("El texto de la miniatura no puede estar vacío.")
    image = Image.fromarray(cv2.cvtColor(frame.frame, cv2.COLOR_BGR2RGB))
    image = image.resize((OUTPUT_WIDTH, OUTPUT_HEIGHT), Image.Resampling.LANCZOS)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    top = int(OUTPUT_HEIGHT * 0.68)
    for y in range(top, OUTPUT_HEIGHT):
        alpha = int(215 * (y - top) / (OUTPUT_HEIGHT - top))
        draw.line((0, y, OUTPUT_WIDTH, y), fill=(7, 12, 24, alpha))
    image = Image.alpha_composite(image.convert("RGBA"), overlay)
    draw = ImageDraw.Draw(image)

    font = _load_thumbnail_font()
    max_width = OUTPUT_WIDTH - 120
    lines = _wrap_title(text, font, max_width=max_width, draw=draw)
    while max(
        draw.textbbox((0, 0), line, font=font, stroke_width=5)[2]
        for line in lines
    ) > max_width and getattr(font, "size", THUMBNAIL_FONT_SIZE) > 36:
        font = _load_thumbnail_font(getattr(font, "size", THUMBNAIL_FONT_SIZE) - 4)
        lines = _wrap_title(text, font, max_width=max_width, draw=draw)
    line_height = int(THUMBNAIL_FONT_SIZE * 1.15)
    block_height = len(lines) * line_height
    y = OUTPUT_HEIGHT - 190 - block_height
    for line in lines:
        bounds = draw.textbbox((0, 0), line, font=font, stroke_width=5)
        x = (OUTPUT_WIDTH - (bounds[2] - bounds[0])) // 2
        draw.text(
            (x, y),
            line,
            font=font,
            fill=(255, 255, 255, 255),
            stroke_width=5,
            stroke_fill=(0, 0, 0, 255),
        )
        y += line_height

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        image.convert("RGB").save(output_path, format="PNG", optimize=True)
    except OSError as exc:
        if getattr(exc, "errno", None) == 28:
            raise ThumbnailError(
                "El disco está lleno. Libera espacio y vuelve a intentarlo."
            ) from exc
        raise ThumbnailError("No se pudo guardar la miniatura.") from exc
    return output_path


def create_thumbnail(
    video_path: Path,
    clip: Clip,
    words: list[TranscriptWord],
    provider: AnalysisProvider,
    output_path: Path,
) -> Path:
    title = generate_thumbnail_text(clip, words, provider)
    frame = select_best_frame(video_path, clip.start, clip.end)
    return compose_thumbnail(frame, title, output_path)


def _load_detectors() -> tuple[cv2.CascadeClassifier, cv2.CascadeClassifier]:
    data_dir = Path(cv2.data.haarcascades)
    face_detector = cv2.CascadeClassifier(
        str(data_dir / "haarcascade_frontalface_default.xml")
    )
    eye_detector = cv2.CascadeClassifier(
        str(data_dir / "haarcascade_eye_tree_eyeglasses.xml")
    )
    if face_detector.empty() or eye_detector.empty():
        raise ThumbnailError("OpenCV no pudo cargar los detectores de rostro y ojos.")
    return face_detector, eye_detector


def _score_frame(
    frame: np.ndarray,
    time: float,
    scale: float,
    face_detector: cv2.CascadeClassifier,
    eye_detector: cv2.CascadeClassifier,
) -> FrameCandidate:
    height, width = frame.shape[:2]
    crop_width, crop_height = calculate_crop_size(width, height)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if scale < 1:
        gray = cv2.resize(
            gray,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )
    face = None
    eye_count = 0
    if len(faces) == 1:
        x, y, face_width, face_height = faces[0]
        center_x = round((x + face_width / 2) / scale)
        center_y = round((y + face_height / 2) / scale)
        face = (center_x, center_y, round(face_width / scale), round(face_height / scale))
        x_scaled, y_scaled, face_width_scaled, face_height_scaled = faces[0]
        face_gray = gray[
            y_scaled : y_scaled + face_height_scaled,
            x_scaled : x_scaled + face_width_scaled,
        ]
        eyes = eye_detector.detectMultiScale(
            face_gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(12, 12),
        )
        eye_count = min(
            sum(1 for _, eye_y, _, _ in eyes if eye_y < face_height_scaled * 0.7),
            2,
        )

    crop_x = (
        min(max(0, face[0] - crop_width // 2), width - crop_width)
        if face
        else max(0, (width - crop_width) // 2)
    )
    crop_y = (
        min(max(0, face[1] - crop_height // 2), height - crop_height)
        if face
        else max(0, (height - crop_height) // 2)
    )
    cropped = frame[crop_y : crop_y + crop_height, crop_x : crop_x + crop_width]
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    face_area = (face[2] * face[3] / (width * height)) if face else 0
    score = math.log1p(sharpness)
    if face:
        score += 12 + min(face_area * 80, 8) + eye_count * 6
    if len(faces) > 1:
        score -= 2
    return FrameCandidate(
        time=time,
        score=score,
        frame=cropped.copy(),
        face=face,
        eyes_visible=eye_count == 2,
    )


def _load_thumbnail_font(
    size: int = THUMBNAIL_FONT_SIZE,
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_candidates = (
        Path("C:/Windows/Fonts/arialbd.ttf"),
        Path("C:/Windows/Fonts/impact.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    )
    for font_path in font_candidates:
        if font_path.is_file():
            return ImageFont.truetype(str(font_path), size)
    return ImageFont.load_default()


def _wrap_title(
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width: int,
    draw: ImageDraw.ImageDraw,
) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and draw.textbbox((0, 0), candidate, font=font)[2] > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines
