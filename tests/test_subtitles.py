import pytest

from app.pipeline.segment import Clip
from app.pipeline.subtitles import build_ass_subtitles
from app.providers.base import TranscriptWord


def test_ass_subtitles_show_one_active_word_at_a_time() -> None:
    clip = Clip(10, 12, "Example", "A test.", 8)
    words = [
        TranscriptWord(9.5, 10.1, "Before"),
        TranscriptWord(10.2, 10.45, "Hello"),
        TranscriptWord(10.5, 10.8, "world!"),
        TranscriptWord(12.0, 12.2, "After"),
    ]

    ass = build_ass_subtitles(clip, words, style="classic", words_per_line=2)
    events = [line for line in ass.splitlines() if line.startswith("Dialogue:")]

    assert len(events) == 3
    assert events[0].startswith("Dialogue: 0,0:00:00.00,0:00:00.10,")
    assert "{\\c&H0000D7FF&}Before{\\c&H00FFFFFF&}" in events[0]
    assert events[1].startswith("Dialogue: 0,0:00:00.20,0:00:00.45,")
    assert "{\\c&H0000D7FF&}Hello{\\c&H00FFFFFF&}" in events[1]
    assert events[2].startswith("Dialogue: 0,0:00:00.50,0:00:00.80,")
    assert "{\\c&H0000D7FF&}world!{\\c&H00FFFFFF&}" in events[2]


def test_ass_subtitles_support_two_styles_and_escape_override_text() -> None:
    clip = Clip(0, 1, "Example", "A test.", 8)
    word = TranscriptWord(0, 0.5, r"{\c&H00000000&}Hi")

    classic = build_ass_subtitles(clip, [word], style="classic")
    impact = build_ass_subtitles(clip, [word], style="impact")

    assert "Style: Default,Arial,54," in classic
    assert "Style: Default,Arial,62," in impact
    assert r"\{\\c&H00000000&\}Hi" in classic


def test_ass_subtitles_reject_unknown_style() -> None:
    with pytest.raises(ValueError, match="Estilo"):
        build_ass_subtitles(Clip(0, 1, "Example", "A test.", 8), [], style="unknown")
