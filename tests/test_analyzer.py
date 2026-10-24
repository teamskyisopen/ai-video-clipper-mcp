"""Unit tests for ViralHighlightDetector, rubrics, and sentence boundary snapping."""

import json
from unittest.mock import MagicMock
import pytest
from pydantic import ValidationError

from core.config import Settings
from domain.models import (
    WordTimestamp,
    TranscriptSegment,
    TranscriptionResult,
    ViralHighlight,
    AnalysisResult,
    HighlightCandidate,
)
from services.analyzer import ViralHighlightDetector, HighlightAnalyzer, VIRALITY_RUBRIC


@pytest.fixture
def mock_transcription_result() -> TranscriptionResult:
    """Mock interview transcript fixture with rich viral candidates."""
    segments = [
        TranscriptSegment(
            id=0,
            text="The single biggest mistake creators make is waiting for motivation.",
            start=0.0,
            end=6.5,
            words=[
                WordTimestamp(word="The", start=0.0, end=0.3),
                WordTimestamp(word="single", start=0.3, end=0.8),
                WordTimestamp(word="biggest", start=0.8, end=1.5),
                WordTimestamp(word="mistake", start=1.5, end=2.2),
                WordTimestamp(word="creators", start=2.2, end=3.0),
                WordTimestamp(word="make", start=3.0, end=3.5),
                WordTimestamp(word="is", start=3.5, end=3.8),
                WordTimestamp(word="waiting", start=3.8, end=4.5),
                WordTimestamp(word="for", start=4.5, end=4.8),
                WordTimestamp(word="motivation.", start=4.8, end=6.5),
            ],
        ),
        TranscriptSegment(
            id=1,
            text="Nobody talks about the brutal friction in the first two minutes. Once you overcome that, momentum carries you.",
            start=6.5,
            end=42.0,
            words=[
                WordTimestamp(word="Nobody", start=6.5, end=7.2),
                WordTimestamp(word="talks", start=7.2, end=7.8),
                WordTimestamp(word="about", start=7.8, end=8.2),
                WordTimestamp(word="the", start=8.2, end=8.4),
                WordTimestamp(word="brutal", start=8.4, end=9.1),
                WordTimestamp(word="friction", start=9.1, end=9.9),
                WordTimestamp(word="you.", start=41.2, end=42.0),
            ],
        ),
        TranscriptSegment(
            id=2,
            text="So stop making excuses and put your work out into the world today.",
            start=42.0,
            end=58.5,
            words=[
                WordTimestamp(word="So", start=42.0, end=42.5),
                WordTimestamp(word="stop", start=42.5, end=43.1),
                WordTimestamp(word="today.", start=57.5, end=58.5),
            ],
        ),
    ]
    return TranscriptionResult(
        video_id="podcast_ep42",
        duration=58.5,
        segments=segments,
        language="en",
        full_text=" ".join(s.text for s in segments),
    )


class TestViralHighlightModels:
    """Validate Pydantic models for viral highlights."""

    def test_viral_highlight_model_success(self) -> None:
        highlight = ViralHighlight(
            hook_title="The Motivation Myth",
            start_time=0.0,
            end_time=42.0,
            virality_score=94,
            reasoning="Strong contrarian hook in first 3 seconds with emotional peak.",
            clean_quote="The single biggest mistake creators make is waiting for motivation.",
        )
        assert highlight.hook_title == "The Motivation Myth"
        assert highlight.start_time == 0.0
        assert highlight.end_time == 42.0
        assert highlight.virality_score == 94
        assert highlight.duration == 42.0

    def test_viral_highlight_score_validation(self) -> None:
        with pytest.raises(ValidationError):
            ViralHighlight(
                hook_title="Invalid",
                start_time=0.0,
                end_time=10.0,
                virality_score=105,  # Exceeds 100
                reasoning="Test",
                clean_quote="Test",
            )

    def test_analysis_result_model(self) -> None:
        result = AnalysisResult(
            highlights=[
                ViralHighlight(
                    hook_title="Hook 1",
                    start_time=0.0,
                    end_time=30.0,
                    virality_score=88,
                    reasoning="Valid reason",
                    clean_quote="Quote",
                )
            ]
        )
        assert len(result.highlights) == 1
        assert result.highlights[0].virality_score == 88


class TestViralHighlightDetector:
    """Validate ViralHighlightDetector prompt formatting, boundary snapping, and structured output parsing."""

    def test_alias_compatibility(self) -> None:
        assert HighlightAnalyzer is ViralHighlightDetector

    def test_rubric_contains_key_elements(self) -> None:
        assert "HOOK STRENGTH (FIRST 3 SECONDS)" in VIRALITY_RUBRIC
        assert "EMOTIONAL PEAKS" in VIRALITY_RUBRIC
        assert "CONTRARIAN STATEMENTS" in VIRALITY_RUBRIC
        assert "CLEAN STORY ARC" in VIRALITY_RUBRIC
        assert "30 to 60 seconds" in VIRALITY_RUBRIC

    def test_snap_to_sentence_boundaries(
        self, mock_transcription_result: TranscriptionResult
    ) -> None:
        detector = ViralHighlightDetector()

        # LLM slightly drifted timestamp (e.g. 0.4s to 41.3s)
        snapped_start, snapped_end = detector.snap_to_sentence_boundaries(
            raw_start=0.4,
            raw_end=41.3,
            transcript=mock_transcription_result,
        )
        # Should snap exactly to segment 0 start (0.0) and segment 1 end (42.0)
        assert snapped_start == 0.0
        assert snapped_end == 42.0

    def test_detect_highlights_with_mock_gemini(
        self, mock_transcription_result: TranscriptionResult
    ) -> None:
        """Verify structured output handling and parsing with mocked Gemini client."""
        mock_payload = {
            "highlights": [
                {
                    "hook_title": "Why Motivation is a Trap",
                    "start_time": 0.2,  # Drifted, will snap to 0.0
                    "end_time": 41.8,   # Drifted, will snap to 42.0
                    "virality_score": 96,
                    "reasoning": "Opens with provocative hook in first 2 seconds, high viewer retention promise.",
                    "clean_quote": "The single biggest mistake creators make is waiting for motivation.",
                }
            ]
        }

        mock_response = MagicMock()
        mock_response.text = json.dumps(mock_payload)

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        detector = ViralHighlightDetector(client=mock_client)
        result = detector.detect_highlights(mock_transcription_result)

        # Verify Gemini client invocation
        assert mock_client.models.generate_content.called
        call_kwargs = mock_client.models.generate_content.call_args[1]
        assert "contents" in call_kwargs
        assert call_kwargs["config"].response_schema == AnalysisResult

        # Verify parsed and snapped result
        assert isinstance(result, AnalysisResult)
        assert len(result.highlights) == 1
        h = result.highlights[0]
        assert h.hook_title == "Why Motivation is a Trap"
        assert h.virality_score == 96
        assert h.start_time == 0.0   # Snapped to sentence start
        assert h.end_time == 42.0    # Snapped to sentence end
        assert h.clean_quote.startswith("The single biggest mistake")

    def test_heuristic_fallback_when_no_api_key(
        self, mock_transcription_result: TranscriptionResult
    ) -> None:
        """Verify that detector falls back to heuristic highlights without error when unconfigured."""
        settings = Settings(gemini_api_key=None, llm_api_key=None)
        detector = ViralHighlightDetector(settings=settings)

        result = detector.detect_highlights(mock_transcription_result)
        assert isinstance(result, AnalysisResult)
        assert len(result.highlights) >= 1
        assert result.highlights[0].start_time == 0.0

    def test_legacy_analyze_adapter(
        self, mock_transcription_result: TranscriptionResult
    ) -> None:
        """Verify backwards compatibility analyze() returns list of HighlightCandidate."""
        settings = Settings(gemini_api_key=None, llm_api_key=None)
        detector = ViralHighlightDetector(settings=settings)
        candidates = detector.analyze(mock_transcription_result)
        assert isinstance(candidates, list)
        assert len(candidates) >= 1
        assert isinstance(candidates[0], HighlightCandidate)
        assert candidates[0].viral_score == 85.0
