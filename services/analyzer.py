"""Viral segment detection service using Gemini and virality rubrics.

Identifies high-retention moments from transcripts and enforces strict Pydantic schemas.
"""

import json
from typing import Any, List, Optional
from loguru import logger

from core.config import Settings, get_settings
from core.exceptions import HighlightAnalysisError
from domain.models import (
    TranscriptionResult,
    TranscriptSegment,
    ViralHighlight,
    AnalysisResult,
    HighlightCandidate,
)

VIRALITY_RUBRIC = """You are an elite short-form video editor specializing in viral TikTok, Reels, and Shorts content.

Evaluate the transcript according to this prioritized virality rubric:
1. HOOK STRENGTH (FIRST 3 SECONDS): The clip MUST open on a high-retention hook that stops scrolling immediately (e.g., curiosity gap, shocking fact, bold challenge, counter-intuitive insight).
2. EMOTIONAL PEAKS: Moments of genuine surprise, raw emotion, humor, tension, or high-energy debate.
3. CONTRARIAN STATEMENTS / OPINION BOMBS: Polarizing, counter-intuitive statements that provoke viewer discussion in the comments.
4. CLEAN STORY ARC: The highlight must feel self-contained with a clear setup and punchline/payoff. NEVER end mid-sentence or mid-thought.
5. DURATION SWEET SPOT: 30 to 60 seconds. Do not exceed 60 seconds or drop below 15 seconds.

CRITICAL TIMESTAMP CONSTRAINT:
- Timestamps must align STRICTLY to complete sentence boundaries.
- Do NOT begin or cut off in the middle of a spoken sentence or phrase.
- Provide the exact start_time and end_time corresponding to the sentence segments provided.
"""


class ViralHighlightDetector:
    """Detects viral highlights using Google Gemini structured output."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        client: Optional[Any] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._client = client

    def _get_gemini_client(self) -> Any:
        """Lazy initialization of Google GenAI client."""
        if self._client is not None:
            return self._client

        api_key = self.settings.gemini_api_key or self.settings.llm_api_key
        if not api_key or api_key.startswith("sk-..."):
            logger.warning("No valid Gemini API key configured")
            return None

        try:
            from google import genai
            self._client = genai.Client(api_key=api_key)
            return self._client
        except Exception as exc:
            logger.error("Failed to initialize Google GenAI client", error=str(exc))
            return None

    def _get_client(self) -> Any:
        """Backwards compatible client getter."""
        return self._get_gemini_client()

    def _call_openai(self, transcript: TranscriptionResult, target_clips: int) -> Optional[AnalysisResult]:
        """Query OpenAI or OpenAI-compatible endpoint using httpx."""
        api_key = self.settings.openai_api_key or self.settings.llm_api_key
        if not api_key or api_key.startswith("sk-..."):
            logger.warning("No valid OpenAI API key configured")
            return None

        base_url = (self.settings.openai_base_url or self.settings.llm_base_url or "https://api.openai.com/v1").rstrip("/")
        model = self.settings.openai_model or self.settings.llm_model

        prompt = (
            f"Video Title / ID: {transcript.video_id}\n"
            f"Total Duration: {transcript.duration:.1f} seconds\n"
            f"Target Highlights Count: {target_clips}\n\n"
            f"Transcript with sentence boundaries:\n"
            f"{self._format_transcript_for_prompt(transcript)}\n\n"
            f"Select the top viral highlights matching all criteria.\n"
            f"Return JSON strictly adhering to schema with 'highlights' list."
        )

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": VIRALITY_RUBRIC},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
        }

        try:
            import httpx
            with httpx.Client(timeout=float(self.settings.analysis_timeout_seconds)) as client:
                resp = client.post(f"{base_url}/chat/completions", headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                return AnalysisResult.model_validate_json(content)
        except Exception as exc:
            logger.error("OpenAI API highlight extraction failed", error=str(exc))
            return None

    def _format_transcript_for_prompt(self, transcript: TranscriptionResult) -> str:
        """Format transcript segments with clear sentence boundaries and timestamps."""
        lines = []
        for seg in transcript.segments:
            lines.append(f"[{seg.start:.2f}s - {seg.end:.2f}s] {seg.text.strip()}")
        return "\n".join(lines)

    def snap_to_sentence_boundaries(
        self,
        raw_start: float,
        raw_end: float,
        transcript: TranscriptionResult,
    ) -> tuple[float, float]:
        """Snap arbitrary LLM timestamps strictly to complete segment/sentence boundaries."""
        if not transcript.segments:
            return raw_start, raw_end

        # Find closest segment start to raw_start
        best_start = transcript.segments[0].start
        min_start_diff = float("inf")
        for seg in transcript.segments:
            diff = abs(seg.start - raw_start)
            if diff < min_start_diff:
                min_start_diff = diff
                best_start = seg.start

        # Find closest segment end to raw_end that is after best_start
        best_end = transcript.segments[-1].end
        min_end_diff = float("inf")
        for seg in transcript.segments:
            if seg.end > best_start:
                diff = abs(seg.end - raw_end)
                if diff < min_end_diff:
                    min_end_diff = diff
                    best_end = seg.end

        # Ensure minimal duration
        if best_end <= best_start:
            best_end = min(best_start + float(self.settings.min_clip_duration_seconds), transcript.duration)

        return round(best_start, 2), round(best_end, 2)

    def detect_highlights(
        self,
        transcript: TranscriptionResult,
        target_clips: int = 3,
    ) -> AnalysisResult:
        """Analyze transcript with LLM structured output to produce AnalysisResult."""
        if not transcript.segments:
            logger.warning("Empty transcript passed to detector")
            return AnalysisResult(highlights=[])

        # 1. Offline Mode / Zero-AI Heuristic bypass
        if self.settings.offline_mode or self.settings.llm_provider.lower() == "none":
            logger.info("Offline mode active; utilizing local speech heuristic analysis")
            return self._detect_heuristically(transcript, target_clips=target_clips)

        # 2. OpenAI / OpenAI-compatible provider
        if self.settings.llm_provider.lower() == "openai":
            logger.info("Querying OpenAI-compatible provider for highlights")
            raw_result = self._call_openai(transcript, target_clips)
            if raw_result and raw_result.highlights:
                snapped_highlights = []
                for h in raw_result.highlights:
                    snapped_start, snapped_end = self.snap_to_sentence_boundaries(
                        h.start_time, h.end_time, transcript
                    )
                    snapped_highlights.append(
                        ViralHighlight(
                            hook_title=h.hook_title,
                            start_time=snapped_start,
                            end_time=snapped_end,
                            virality_score=h.virality_score,
                            reasoning=h.reasoning,
                            clean_quote=h.clean_quote,
                        )
                    )
                logger.info("Detected viral highlights via OpenAI", count=len(snapped_highlights))
                return AnalysisResult(highlights=snapped_highlights)
            logger.warning("OpenAI analysis returned empty or failed, falling back to heuristic")
            return self._detect_heuristically(transcript, target_clips=target_clips)

        # 3. Google Gemini provider
        client = self._get_gemini_client()
        if client is not None:
            try:
                from google.genai import types

                prompt = (
                    f"Video Title / ID: {transcript.video_id}\n"
                    f"Total Duration: {transcript.duration:.1f} seconds\n"
                    f"Target Highlights Count: {target_clips}\n\n"
                    f"Transcript with sentence boundaries:\n"
                    f"{self._format_transcript_for_prompt(transcript)}\n\n"
                    f"Select the top viral highlights matching all criteria."
                )

                config = types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=AnalysisResult,
                    system_instruction=VIRALITY_RUBRIC,
                    temperature=0.2,
                )

                response = client.models.generate_content(
                    model=self.settings.gemini_model,
                    contents=prompt,
                    config=config,
                )

                # Validate structured response
                raw_text = response.text if hasattr(response, "text") else str(response)
                result = AnalysisResult.model_validate_json(raw_text)

                # Snap timestamps to strict sentence boundaries
                snapped_highlights = []
                for h in result.highlights:
                    snapped_start, snapped_end = self.snap_to_sentence_boundaries(
                        h.start_time, h.end_time, transcript
                    )
                    snapped_highlights.append(
                        ViralHighlight(
                            hook_title=h.hook_title,
                            start_time=snapped_start,
                            end_time=snapped_end,
                            virality_score=h.virality_score,
                            reasoning=h.reasoning,
                            clean_quote=h.clean_quote,
                        )
                    )

                logger.info("Detected viral highlights via Gemini", count=len(snapped_highlights))
                return AnalysisResult(highlights=snapped_highlights)

            except Exception as exc:
                logger.error("Gemini API highlight extraction failed, using heuristic fallback", error=str(exc))

        # Heuristic fallback if no API key or API call failed
        return self._detect_heuristically(transcript, target_clips=target_clips)

    def _detect_heuristically(
        self,
        transcript: TranscriptionResult,
        target_clips: int = 1,
    ) -> AnalysisResult:
        """Heuristic fallback to identify highlight candidates based on sentence boundaries."""
        if not transcript.segments:
            return AnalysisResult(highlights=[])

        highlights: List[ViralHighlight] = []

        # 1. Primary opening hook (proven high-retention starting segment)
        first_seg = transcript.segments[0]
        end_time = min(first_seg.start + float(self.settings.max_clip_duration_seconds), transcript.duration)
        snapped_start, snapped_end = self.snap_to_sentence_boundaries(first_seg.start, end_time, transcript)

        highlights.append(
            ViralHighlight(
                hook_title="Opening Statement Hook",
                start_time=snapped_start,
                end_time=snapped_end,
                virality_score=85,
                reasoning="Opening thesis with immediate context setup.",
                clean_quote=first_seg.text,
            )
        )

        # 2. If user requested more clips, extract additional candidate windows across the duration
        if target_clips > 1 and len(transcript.segments) > 2:
            step = max(1, len(transcript.segments) // target_clips)
            for i in range(1, target_clips):
                seg_idx = min(i * step, len(transcript.segments) - 1)
                seg = transcript.segments[seg_idx]
                h_end = min(seg.start + float(self.settings.max_clip_duration_seconds), transcript.duration)
                s_start, s_end = self.snap_to_sentence_boundaries(seg.start, h_end, transcript)
                if s_start >= s_end or any(abs(h.start_time - s_start) < 10.0 for h in highlights):
                    continue
                score = max(60, 85 - i * 5)
                highlights.append(
                    ViralHighlight(
                        hook_title=f"Spoken Segment Highlight #{i + 1}",
                        start_time=s_start,
                        end_time=s_end,
                        virality_score=score,
                        reasoning=f"High-density speech segment with complete sentence boundary.",
                        clean_quote=seg.text,
                    )
                )

        return AnalysisResult(highlights=highlights)

    def analyze(
        self,
        transcript: TranscriptionResult,
        min_duration: Optional[int] = None,
        max_duration: Optional[int] = None,
    ) -> List[HighlightCandidate]:
        """Compatibility adapter returning legacy HighlightCandidate objects."""
        result = self.detect_highlights(transcript)
        candidates = []
        for idx, h in enumerate(result.highlights):
            candidates.append(
                HighlightCandidate(
                    clip_id=f"clip_{idx + 1:03d}",
                    start_time=h.start_time,
                    end_time=h.end_time,
                    title=h.hook_title,
                    hook=h.clean_quote[:80],
                    summary=h.clean_quote,
                    viral_score=float(h.virality_score),
                    rationale=h.reasoning,
                )
            )
        return candidates


# Backwards compatibility alias
HighlightAnalyzer = ViralHighlightDetector
