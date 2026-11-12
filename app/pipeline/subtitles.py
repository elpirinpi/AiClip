import re
from dataclasses import dataclass
from pathlib import Path

from app.pipeline.segment import Clip
from app.providers.base import TranscriptWord


@dataclass(frozen=True)
class SubtitleStyle:
    font_name: str
    font_size: int
    primary_color: str
    outline_color: str
    highlight_color: str
    bold: bool
    outline: int
    shadow: int
    vertical_margin: int = 72


SUBTITLE_STYLES = {
    "classic": SubtitleStyle(
        font_name="Arial",
        font_size=54,
        primary_color="&H00FFFFFF",
        outline_color="&H00101010",
        highlight_color="&H0000D7FF",
        bold=True,
        outline=3,
        shadow=1,
    ),
    "impact": SubtitleStyle(
        font_name="Arial",
        font_size=62,
        primary_color="&H00FFFFFF",
        outline_color="&H00000000",
        highlight_color="&H0000FFFF",
        bold=True,
        outline=4,
        shadow=2,
    ),
}


def build_ass_subtitles(
    clip: Clip,
    words: list[TranscriptWord],
    style: str = "classic",
    words_per_line: int = 6,
) -> str:
    if style not in SUBTITLE_STYLES:
        raise ValueError(f"Estilo de subtítulos no válido: {style}")
    if words_per_line < 1:
        raise ValueError("El número de palabras por línea debe ser al menos uno.")

    subtitle_style = SUBTITLE_STYLES[style]
    script = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        "YCbCr Matrix: TV.601",
        "PlayResX: 1080",
        "PlayResY: 1920",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding",
        (
            f"Style: Default,{subtitle_style.font_name},{subtitle_style.font_size},"
            f"{subtitle_style.primary_color},{subtitle_style.highlight_color},"
            f"{subtitle_style.outline_color},&H80000000,"
            f"{-1 if subtitle_style.bold else 0},0,0,0,100,100,0,0,1,"
            f"{subtitle_style.outline},{subtitle_style.shadow},2,80,80,"
            f"{subtitle_style.vertical_margin},1"
        ),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
        "Effect, Text",
    ]

    clip_words = [
        word
        for word in sorted(words, key=lambda item: (item.start, item.end))
        if word.text.strip() and word.end > clip.start and word.start < clip.end
    ]
    for group_start in range(0, len(clip_words), words_per_line):
        group = clip_words[group_start : group_start + words_per_line]
        for active_index, word in enumerate(group):
            start = max(word.start, clip.start) - clip.start
            end = min(word.end, clip.end) - clip.start
            if end <= start:
                continue
            text = " ".join(
                _format_word(item.text, index == active_index, subtitle_style)
                for index, item in enumerate(group)
            )
            script.append(
                f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},"
                f"Default,,0,0,0,,{text}"
            )
    return "\n".join(script) + "\n"


def write_ass_subtitles(
    path: Path,
    clip: Clip,
    words: list[TranscriptWord],
    style: str = "classic",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        build_ass_subtitles(clip, words, style),
        encoding="utf-8-sig",
        newline="\n",
    )


def _format_word(text: str, active: bool, style: SubtitleStyle) -> str:
    escaped = _escape_ass_text(text)
    if active:
        return (
            f"{{\\c{style.highlight_color}&}}{escaped}"
            f"{{\\c{style.primary_color}&}}"
        )
    return escaped


def _escape_ass_text(text: str) -> str:
    text = text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
    return re.sub(r"[\r\n]+", " ", text)


def _ass_time(seconds: float) -> str:
    centiseconds = max(0, round(seconds * 100))
    hours, remainder = divmod(centiseconds, 360_000)
    minutes, remainder = divmod(remainder, 6_000)
    seconds, centiseconds = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"
