"""End-to-end integration test executing the complete video clipping pipeline on a live video.

Measures and logs execution times for:
1. Audio extraction
2. Faster-Whisper transcription
3. Gemini / Rubric viral moment selection
4. Video slice retrieval
5. Smart framing & speaker tracking
6. ASS karaoke subtitle generation
7. FFmpeg composition to 9:16 portrait MP4
"""

import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Dict, List, Optional
from loguru import logger
import pytest

# Ensure project root is in sys.path when run directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import get_settings
from domain.models import AnalysisResult, ClipResult, TranscriptionResult
from services.coordinator import PipelineCoordinator
from services.downloader import MediaDownloader
from services.transcriber import FasterWhisperTranscriber
from services.analyzer import ViralHighlightDetector
from services.framer import SmartVideoFramer
from services.subtitles import SubtitleGenerator
from services.compositor import FFmpegCompositor

# Short public YouTube video: "Me at the zoo" (19 seconds, 0:19)
TEST_YOUTUBE_URL = os.environ.get("TEST_VIDEO_URL", "https://www.youtube.com/watch?v=jNQXAC9IVRw")


async def run_pipeline_e2e(
    url: str = TEST_YOUTUBE_URL,
    output_dir: Optional[Path] = None,
) -> ClipResult:
    """Execute all pipeline stages step-by-step with high-resolution execution timing."""
    settings = get_settings()
    out_dir = output_dir or settings.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    downloader = MediaDownloader(settings)
    transcriber = FasterWhisperTranscriber(settings)
    analyzer = ViralHighlightDetector(settings)
    framer = SmartVideoFramer(settings)
    subtitles = SubtitleGenerator(settings)
    compositor = FFmpegCompositor(settings)

    total_start = time.perf_counter()
    timings = {}

    logger.info("=" * 70)
    logger.info("STARTING END-TO-END PIPELINE INTEGRATION TEST")
    logger.info(f"Target Video URL: {url}")
    logger.info("=" * 70)

    with settings.create_managed_temp_dir(prefix="e2e_") as temp_dir:
        # ---------------------------------------------------------------------
        # Step 1: Audio Extraction
        # ---------------------------------------------------------------------
        t0 = time.perf_counter()
        logger.info("[Step 1/7] Extracting fast 128kbps audio...")
        audio_path = await asyncio.to_thread(downloader.download_audio, url, temp_dir)
        timings["audio_extraction_seconds"] = round(time.perf_counter() - t0, 3)
        logger.info(f"--> Audio Extracted: {audio_path.name} in {timings['audio_extraction_seconds']}s")

        # ---------------------------------------------------------------------
        # Step 2: Transcription with Word Timestamps
        # ---------------------------------------------------------------------
        t0 = time.perf_counter()
        logger.info("[Step 2/7] Transcribing speech with Faster-Whisper...")
        transcript = await asyncio.to_thread(transcriber.transcribe, audio_path)
        timings["transcription_seconds"] = round(time.perf_counter() - t0, 3)
        logger.info(
            f"--> Transcribed in {timings['transcription_seconds']}s | "
            f"Duration: {transcript.duration:.2f}s | "
            f"Segments: {len(transcript.segments)} | "
            f"Text: '{transcript.full_text[:60]}...'"
        )

        # ---------------------------------------------------------------------
        # Step 3: Viral Moment Selection
        # ---------------------------------------------------------------------
        t0 = time.perf_counter()
        logger.info("[Step 3/7] Selecting viral highlight moments via rubric...")
        analysis = await asyncio.to_thread(analyzer.detect_highlights, transcript, target_clips=1)
        timings["viral_analysis_seconds"] = round(time.perf_counter() - t0, 3)

        assert len(analysis.highlights) > 0, "No viral highlights detected"
        highlight = analysis.highlights[0]
        timings["highlight_duration_seconds"] = round(highlight.duration, 2)
        logger.info(
            f"--> Highlight Selected in {timings['viral_analysis_seconds']}s | "
            f"Title: '{highlight.hook_title}' | "
            f"Score: {highlight.virality_score}/100 | "
            f"Window: [{highlight.start_time:.2f}s - {highlight.end_time:.2f}s]"
        )

        # ---------------------------------------------------------------------
        # Step 4: Video Slice Retrieval
        # ---------------------------------------------------------------------
        t0 = time.perf_counter()
        logger.info(
            f"[Step 4/7] Downloading video slice [{highlight.start_time:.1f}s - {highlight.end_time:.1f}s]..."
        )
        slice_path = await asyncio.to_thread(
            downloader.download_video_slice,
            url,
            highlight.start_time,
            highlight.end_time,
            temp_dir,
        )
        timings["video_slice_download_seconds"] = round(time.perf_counter() - t0, 3)
        logger.info(f"--> Video Slice Retrieved: {slice_path.name} in {timings['video_slice_download_seconds']}s")

        # ---------------------------------------------------------------------
        # Step 5: Reframing & Speaker Tracking
        # ---------------------------------------------------------------------
        t0 = time.perf_counter()
        logger.info("[Step 5/7] Calculating 9:16 portrait framing and speaker tracking...")
        clip_duration = max(0.1, highlight.end_time - highlight.start_time)
        # Probe actual resolution of the downloaded slice
        probe_cmd = [
            str(settings.ffprobe_path),
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "json",
            str(slice_path),
        ]
        probe_res = subprocess.run(probe_cmd, stdout=subprocess.PIPE, text=True)
        slice_meta = json.loads(probe_res.stdout) if probe_res.stdout else {}
        s_streams = slice_meta.get("streams", [])
        source_w = int(s_streams[0]["width"]) if s_streams else 1920
        source_h = int(s_streams[0]["height"]) if s_streams else 1080

        coords = await asyncio.to_thread(
            framer.calculate_framing_trajectory,
            video_path=slice_path,
            start_time=0.0,
            end_time=clip_duration,
            source_width=source_w,
            source_height=source_h,
        )
        timings["reframing_calc_seconds"] = round(time.perf_counter() - t0, 3)

        primary_coord = coords[0] if coords else None
        crop_x = primary_coord.crop_x if primary_coord else 0
        crop_y = primary_coord.crop_y if primary_coord else 0
        crop_w = primary_coord.width if primary_coord else 608
        crop_h = primary_coord.height if primary_coord else 1080
        logger.info(
            f"--> Reframing Calculated in {timings['reframing_calc_seconds']}s | "
            f"Coordinates: crop={crop_w}x{crop_h} at ({crop_x}, {crop_y})"
        )

        # ---------------------------------------------------------------------
        # Step 6: Subtitle Generation
        # ---------------------------------------------------------------------
        t0 = time.perf_counter()
        logger.info("[Step 6/7] Generating animated karaoke ASS subtitles...")
        ass_path = temp_dir / "e2e_subtitles.ass"
        await asyncio.to_thread(
            subtitles.write_ass_file,
            output_path=ass_path,
            segments=transcript.segments,
            clip_start=highlight.start_time,
            clip_end=highlight.end_time,
            is_split_screen=any(c.is_split for c in coords),
        )
        timings["subtitle_gen_seconds"] = round(time.perf_counter() - t0, 3)
        logger.info(f"--> Subtitles Generated: {ass_path.name} in {timings['subtitle_gen_seconds']}s")

        # ---------------------------------------------------------------------
        # Step 7: FFmpeg Rendering & Composition
        # ---------------------------------------------------------------------
        t0 = time.perf_counter()
        logger.info("[Step 7/7] Compositing final 9:16 portrait video via FFmpeg...")
        final_mp4 = out_dir / "e2e_final_short_clip.mp4"

        clip_result = await compositor.render_video(
            input_video_path=slice_path,
            output_path=final_mp4,
            start_time=0.0,
            end_time=clip_duration,
            crop_x=crop_x,
            crop_y=crop_y,
            crop_width=crop_w,
            crop_height=crop_h,
            subtitle_ass_path=ass_path,
            target_width=1080,
            target_height=1920,
            target_fps=30,
            clip_id="e2e_test_clip",
        )
        timings["ffmpeg_render_seconds"] = round(time.perf_counter() - t0, 3)
        timings["total_pipeline_seconds"] = round(time.perf_counter() - total_start, 3)

        logger.info("=" * 70)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info(f"Final MP4 Output Path: {clip_result.output_path}")
        logger.info(f"File Size: {clip_result.file_size_bytes:,} bytes")
        logger.info("EXECUTION TIME BREAKDOWN:")
        for step, dur in timings.items():
            logger.info(f"  - {step}: {dur}s")
        logger.info("=" * 70)

        return clip_result


def verify_rendered_video(mp4_path: Path) -> dict:
    """Probe rendered output file with ffprobe and verify technical specifications."""
    assert mp4_path.is_file(), f"Output file does not exist: {mp4_path}"
    assert mp4_path.stat().st_size > 0, "Output file is empty (0 bytes)"

    settings = get_settings()
    ffprobe_bin = str(settings.ffprobe_path)

    cmd = [
        ffprobe_bin,
        "-v", "error",
        "-show_entries", "stream=width,height,codec_name,codec_type,pix_fmt:format=duration,size",
        "-of", "json",
        str(mp4_path),
    ]

    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert res.returncode == 0, f"ffprobe execution failed: {res.stderr}"

    metadata = json.loads(res.stdout)
    streams = metadata.get("streams", [])

    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]

    assert len(video_streams) > 0, "No video stream found in rendered MP4"
    assert len(audio_streams) > 0, "No audio stream found in rendered MP4"

    v_stream = video_streams[0]
    width = int(v_stream["width"])
    height = int(v_stream["height"])

    # Strict Invariant: 9:16 Portrait Aspect Ratio
    assert width == 1080, f"Expected width 1080, got {width}"
    assert height == 1920, f"Expected height 1920, got {height}"
    assert (width / height) == (9 / 16), f"Aspect ratio is not 9:16: {width}/{height}"

    return {
        "width": width,
        "height": height,
        "video_codec": v_stream.get("codec_name"),
        "audio_codec": audio_streams[0].get("codec_name"),
        "file_size": int(metadata.get("format", {}).get("size", 0)),
        "duration": float(metadata.get("format", {}).get("duration", 0.0)),
    }


def test_pipeline_end_to_end(tmp_path: Path) -> None:
    """Pytest test case executing the full e2e pipeline and asserting technical invariants."""
    clip_result = asyncio.run(run_pipeline_e2e(output_dir=tmp_path))
    specs = verify_rendered_video(clip_result.output_path)

    assert specs["width"] == 1080
    assert specs["height"] == 1920
    assert specs["file_size"] > 0
    assert specs["video_codec"] == "h264"
    assert specs["audio_codec"] == "aac"


if __name__ == "__main__":
    result = asyncio.run(run_pipeline_e2e())
    probe_info = verify_rendered_video(result.output_path)
    print("\nVerified Technical Specifications:")
    print(json.dumps(probe_info, indent=2))
