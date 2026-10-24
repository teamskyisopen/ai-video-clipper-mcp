"""Pipeline coordinator orchestrating end-to-end media download, transcription,

analysis, framing, subtitle styling, and FFmpeg video compositing.
"""

import asyncio
from pathlib import Path
import time
from typing import List, Optional
import uuid
from loguru import logger

from core.config import Settings, get_settings
from core.exceptions import VideoClipperError
from domain.models import (
    AnalysisResult,
    ClipResult,
    FramingCoordinate,
    TranscriptSegment,
    TranscriptionResult,
    ViralHighlight,
    WordTimestamp,
)
from services.downloader import MediaDownloader
from services.transcriber import FasterWhisperTranscriber
from services.analyzer import ViralHighlightDetector
from services.framer import SmartVideoFramer
from services.subtitles import SubtitleGenerator
from services.compositor import FFmpegCompositor


class PipelineCoordinator:
    """Coordinates the modular pipeline services with automatic temp cleanup."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        downloader: Optional[MediaDownloader] = None,
        transcriber: Optional[FasterWhisperTranscriber] = None,
        analyzer: Optional[ViralHighlightDetector] = None,
        framer: Optional[SmartVideoFramer] = None,
        subtitles: Optional[SubtitleGenerator] = None,
        compositor: Optional[FFmpegCompositor] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.downloader = downloader or MediaDownloader(self.settings)
        self.transcriber = transcriber or FasterWhisperTranscriber(self.settings)
        self.analyzer = analyzer or ViralHighlightDetector(self.settings)
        self.framer = framer or SmartVideoFramer(self.settings)
        self.subtitles = subtitles or SubtitleGenerator(self.settings)
        self.compositor = compositor or FFmpegCompositor(self.settings)

    async def analyze_video_highlights(
        self,
        url: str,
        max_clips: int = 3,
    ) -> AnalysisResult:
        """Download fast audio, transcribe, analyze highlights, and clean up temp audio."""
        logger.info("Pipeline: Starting highlight analysis", url=url, max_clips=max_clips)
        start_time = time.time()

        with self.settings.create_managed_temp_dir(prefix="analyze_") as temp_dir:
            # Step 1: Download fast audio (MP3/AAC 128kbps)
            audio_path = await asyncio.to_thread(self.downloader.download_audio, url, temp_dir)
            logger.info("Pipeline: Audio downloaded", path=str(audio_path), elapsed=f"{time.time() - start_time:.2f}s")

            # Step 2: Transcribe speech with word-level timestamps
            transcript = await asyncio.to_thread(self.transcriber.transcribe, audio_path)
            logger.info(
                "Pipeline: Transcription complete",
                duration=transcript.duration,
                segments=len(transcript.segments),
                elapsed=f"{time.time() - start_time:.2f}s",
            )

            # Step 3: Analyze highlights via Gemini virality rubric
            analysis = await asyncio.to_thread(
                self.analyzer.detect_highlights, transcript, target_clips=max_clips
            )
            logger.info(
                "Pipeline: Highlight analysis complete",
                highlights_found=len(analysis.highlights),
                total_time=f"{time.time() - start_time:.2f}s",
            )
            return analysis

    async def render_highlight_clip(
        self,
        url: str,
        start_time: float,
        end_time: float,
        subtitle_preset: str = "karaoke_yellow",
        clip_id: Optional[str] = None,
        transcript_segments: Optional[List[TranscriptSegment]] = None,
    ) -> ClipResult:
        """Download video slice, compute framing, generate animated subtitles, and composite."""
        clip_uid = clip_id or f"clip_{uuid.uuid4().hex[:8]}"
        duration = max(0.1, end_time - start_time)
        logger.info(
            "Pipeline: Starting clip rendering",
            clip_id=clip_uid,
            url=url,
            start=start_time,
            end=end_time,
            duration=duration,
        )

        with self.settings.create_managed_temp_dir(prefix="render_") as temp_dir:
            # 1. Fetch only the video slice
            slice_path = await asyncio.to_thread(
                self.downloader.download_video_slice,
                url,
                start_time,
                end_time,
                temp_dir,
            )

            # 2. Compute dynamic portrait crop & speaker tracking
            import cv2
            cap = cv2.VideoCapture(str(slice_path))
            src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
            src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
            cap.release()

            coords = await asyncio.to_thread(
                self.framer.calculate_framing_trajectory,
                video_path=slice_path,
                start_time=0.0,
                end_time=duration,
                source_width=src_w,
                source_height=src_h,
            )

            primary_coord = coords[0] if coords else None
            is_split = any(c.is_split for c in coords)
            crop_x = primary_coord.crop_x if primary_coord else None
            crop_y = primary_coord.crop_y if primary_coord else None
            crop_w = primary_coord.width if primary_coord else None
            crop_h = primary_coord.height if primary_coord else None

            # 3. Transcribe audio from slice if segments not provided
            if transcript_segments is None:
                try:
                    slice_transcript = await asyncio.to_thread(self.transcriber.transcribe, slice_path)
                    segments_to_render = slice_transcript.segments
                except Exception as exc:
                    logger.warning("Slice transcription failed, falling back to dummy subtitle", error=str(exc))
                    segments_to_render = [
                        TranscriptSegment(
                            id=0,
                            text="Highlight Clip",
                            start=0.0,
                            end=duration,
                            words=[WordTimestamp(word="Highlight", start=0.0, end=duration / 2.0)],
                        )
                    ]
            else:
                segments_to_render = transcript_segments

            # 4. Generate ASS subtitles with karaoke animations
            ass_path = temp_dir / f"{clip_uid}.ass"
            await asyncio.to_thread(
                self.subtitles.write_ass_file,
                output_path=ass_path,
                segments=segments_to_render,
                clip_start=0.0,
                clip_end=duration,
                is_split_screen=is_split,
            )

            # 5. Composite final video
            self.settings.output_dir.mkdir(parents=True, exist_ok=True)
            output_mp4 = self.settings.output_dir / f"{clip_uid}.mp4"

            clip_result = await self.compositor.render_video(
                input_video_path=slice_path,
                output_path=output_mp4,
                start_time=0.0,
                end_time=duration,
                crop_x=crop_x,
                crop_y=crop_y,
                crop_width=crop_w,
                crop_height=crop_h,
                subtitle_ass_path=ass_path,
                clip_id=clip_uid,
            )

            return clip_result

    async def auto_clip_video(
        self,
        url: str,
        clip_count: int = 3,
    ) -> List[ClipResult]:
        """High-level macro tool: Analyzes highlights and renders all top clips end-to-end."""
        logger.info("Pipeline: Starting auto_clip_video end-to-end", url=url, clip_count=clip_count)
        analysis = await self.analyze_video_highlights(url=url, max_clips=clip_count)

        results: List[ClipResult] = []
        for idx, highlight in enumerate(analysis.highlights[:clip_count]):
            clip_id = f"autoclip_{idx + 1:02d}_{int(highlight.start_time)}_{int(highlight.end_time)}"
            logger.info("Pipeline: Rendering detected highlight", clip_id=clip_id, title=highlight.hook_title)

            try:
                clip_res = await self.render_highlight_clip(
                    url=url,
                    start_time=highlight.start_time,
                    end_time=highlight.end_time,
                    clip_id=clip_id,
                )
                results.append(clip_res)
            except Exception as exc:
                logger.error("Pipeline: Failed rendering clip", clip_id=clip_id, error=str(exc))

        return results
