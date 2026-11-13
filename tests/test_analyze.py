import json
import re

import pytest

from app.pipeline.analyze import (
    AnalysisError,
    _generate_candidates,
    analyze_transcript,
    build_highlights_prompt,
)
from app.providers.base import RequestTooLargeError, TranscriptWord


def _transcript(duration: int) -> list[TranscriptWord]:
    words = []
    for index in range(duration * 2):
        second = index * 0.5
        text = f"palabra{index}"
        if index % 10 == 9:
            text += "."
        words.append(TranscriptWord(second, second + 0.4, text))
    return words


def _selection(candidate_id: int = 1) -> str:
    return json.dumps(
        {
            "clips": [
                {
                    "candidate_id": candidate_id,
                    "title": "A complete clip title",
                    "reason": "It makes sense on its own.",
                    "score": 9,
                }
            ]
        }
    )


class StaticProvider:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompt = ""

    def generate(self, prompt: str) -> str:
        self.prompt = prompt
        return self.response


def test_local_candidate_generation_is_fast_bounded_and_non_overlapping() -> None:
    candidates = _generate_candidates(_transcript(1200), target_clips=7)

    assert 6 <= len(candidates) <= 14
    assert all(30 <= candidate.end - candidate.start <= 90 for candidate in candidates)
    assert all(
        (left.start, left.end) != (right.start, right.end)
        for left, right in zip(candidates, candidates[1:])
    )
    assert all(
        max(0, min(left.end, right.end) - max(left.start, right.start))
        / min(left.end - left.start, right.end - right.start)
        == 0
        for index, left in enumerate(candidates)
        for right in candidates[index + 1 :]
    )
    assert candidates == sorted(candidates, key=lambda candidate: candidate.start)
    assert candidates[0].start < 100
    assert candidates[-1].end > 1000


def test_highlights_prompt_contains_compact_candidates_and_clip_target() -> None:
    candidates = _generate_candidates(_transcript(300), target_clips=2)
    prompt = build_highlights_prompt(candidates, target_clips=2)

    assert "Intenta seleccionar hasta 2 clips" in prompt
    assert "[CANDIDATO 1 |" in prompt
    assert "TRANSCRIPCIÓN COMPLETA" not in prompt
    assert candidates[0].text in prompt


def test_analyze_transcript_maps_selection_to_exact_candidate_timestamps() -> None:
    words = _transcript(180)
    candidates = _generate_candidates(words, target_clips=1)
    provider = StaticProvider(_selection(candidates[0].candidate_id))
    progress = []

    clips = analyze_transcript(words, provider, progress_callback=progress.append)

    assert len(clips) == 1
    assert clips[0].start == candidates[0].start
    assert clips[0].end == candidates[0].end
    assert clips[0].title == "A complete clip title"
    assert clips[0].score == 9
    assert "CANDIDATO" in provider.prompt
    assert "Intenta seleccionar hasta 1 clip" in provider.prompt
    assert progress == [76, 80, 90]


@pytest.mark.parametrize(
    "response",
    [
        "not json",
        "{}",
        '{"clips": [{"candidate_id": 99, "title": "No", "reason": "Unknown", "score": 8}]}',
        '{"clips": [{"candidate_id": 1, "title": "", "reason": "Missing title", "score": 8}]}',
    ],
)
def test_analyze_transcript_reports_invalid_provider_responses(response: str) -> None:
    with pytest.raises(AnalysisError):
        analyze_transcript(_transcript(120), StaticProvider(response))


def test_short_transcript_with_no_complete_clip_skips_api_call() -> None:
    class FailingProvider:
        def generate(self, _prompt: str) -> str:
            raise AssertionError("Should not call the analysis provider.")

    assert analyze_transcript(_transcript(20), FailingProvider()) == []


def test_oversized_candidate_batch_is_split_and_ranked() -> None:
    class SizeLimitedProvider:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, prompt: str) -> str:
            self.calls += 1
            candidate_id = int(re.search(r"\[CANDIDATO (\d+)", prompt).group(1))
            if prompt.count("[CANDIDATO") > 2:
                raise RequestTooLargeError("request too large")
            return _selection(candidate_id)

    provider = SizeLimitedProvider()
    clips = analyze_transcript(_transcript(600), provider)

    assert provider.calls > 1
    assert clips
    assert len(clips) == 4


def test_single_candidate_too_large_has_a_clear_error() -> None:
    class AlwaysTooLargeProvider:
        def generate(self, _prompt: str) -> str:
            raise RequestTooLargeError("request too large")

    with pytest.raises(RequestTooLargeError, match="candidato individual"):
        analyze_transcript(_transcript(120), AlwaysTooLargeProvider())
