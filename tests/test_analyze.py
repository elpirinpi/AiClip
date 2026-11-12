import json

import pytest

from app.pipeline.analyze import AnalysisError, analyze_transcript
from app.providers.base import TranscriptWord


class StaticProvider:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompt = ""

    def generate(self, prompt: str) -> str:
        self.prompt = prompt
        return self.response


def test_analyze_transcript_validates_json_and_limits_title_words() -> None:
    response = json.dumps(
        [
            {
                "start": 10,
                "end": 45,
                "title": "one two three four five six seven eight nine",
                "reason": "The example is self-contained.",
                "score": 8,
            }
        ]
    )
    provider = StaticProvider(response)

    clips = analyze_transcript(
        [TranscriptWord(0, 1, "Hello.")],
        provider,
    )

    assert len(clips[0].title.split()) == 8
    assert "[0.00-1.00] Hello." in provider.prompt
    assert clips[0].score == 8


@pytest.mark.parametrize("response", ["not json", "{}", '[{"start": 0}]'])
def test_analyze_transcript_reports_invalid_provider_responses(response: str) -> None:
    with pytest.raises(AnalysisError):
        analyze_transcript([TranscriptWord(0, 1, "Hello.")], StaticProvider(response))
