from app.pipeline.segment import ClipCandidate, snap_clips_to_sentences
from app.providers.base import TranscriptWord


def _words() -> list[TranscriptWord]:
    return [
        TranscriptWord(0, 12, "Opening."),
        TranscriptWord(15, 35, "First story."),
        TranscriptWord(40, 65, "Second story."),
        TranscriptWord(70, 100, "Closing."),
    ]


def test_clips_snap_to_sentence_boundaries_and_sort_by_score() -> None:
    candidates = [
        ClipCandidate(30, 60, "Lower", "A reason.", 6),
        ClipCandidate(35, 62, "Higher", "A reason.", 9),
    ]

    clips, warnings = snap_clips_to_sentences(candidates, _words())

    assert warnings == ["Se descartó «Lower» porque se solapa con otro clip."]
    assert len(clips) == 1
    assert (clips[0].start, clips[0].end) == (15, 65)
    assert clips[0].score == 9


def test_clips_outside_duration_limits_are_dropped() -> None:
    candidates = [ClipCandidate(43, 60, "Short", "A reason.", 9)]

    clips, warnings = snap_clips_to_sentences(candidates, _words())

    assert clips == []
    assert warnings and "duraría" in warnings[0]


def test_no_candidate_clips_returns_empty_result() -> None:
    assert snap_clips_to_sentences([], _words()) == ([], [])
