"""Integration tests for subtitle generation and FFmpeg video compositing."""

import json
from pathlib import Path
import subprocess
import pytest

from core.config import Settings
from domain.models import TranscriptSegment, WordTimestamp
from services.subtitles import SubtitleGenerator, format_ass_timestamp
from services.compositor import FFmpegCompositor, VideoCompositor


@pytest.fixture
def dummy_color_bars_video(tmp_path: Path) -> Path:
    """Generate a 5-second 1920x1080 30fps SMPTE color bars video with audio."""
    settings = Settings()
    source_video = tmp_path / "color_bars_16x9.mp4"

    cmd = [
        str(settings.ffmpeg_path),
        "-y",
        "-f", "lavfi",
        "-i", "smptebars=duration=5:size=1920x1080:rate=30",
        "-f", "lavfi",
        "-i", "sine=frequency=440:duration=5",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        str(source_video),
    ]

    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert res.returncode == 0, f"FFmpeg failed to create test video: {res.stderr}"
    assert source_video.is_file()
    assert source_video.stat().st_size > 0
    return source_video


class TestSubtitleGenerator:
    """Validate ASS script generation, karaoke formatting, and safe positioning."""

    def test_format_ass_timestamp(self) -> None:
        assert format_ass_timestamp(0.0) == "0:00:00.00"
        assert format_ass_timestamp(1.25) == "0:00:01.25"
        assert format_ass_timestamp(65.5) == "0:01:05.50"
        assert format_ass_timestamp(3661.05) == "1:01:01.05"

    def test_generate_ass_header_styling(self) -> None:
        settings = Settings(
            subtitle_font_name="Arial",
            subtitle_primary_color="&H00FFFFFF&",
            subtitle_highlight_color="&H0000FFFF&",
            subtitle_margin_v=240,
        )
        generator = SubtitleGenerator(settings)
        header = generator.generate_ass_header(is_split_screen=False)

        assert "[Script Info]" in header
        assert "PlayResX: 1080" in header
        assert "PlayResY: 1920" in header
        assert "Style: Default,Arial" in header
        assert "&H00FFFFFF&" in header
        assert ",240,1" in header  # MarginV = 240

    def test_karaoke_active_word_highlighting(self, tmp_path: Path) -> None:
        settings = Settings(
            subtitle_highlight_color="&H0000FFFF&",
            subtitle_primary_color="&H00FFFFFF&",
        )
        generator = SubtitleGenerator(settings)

        words = [
            WordTimestamp(word="Stop", start=0.0, end=0.8),
            WordTimestamp(word="scrolling", start=0.8, end=1.8),
            WordTimestamp(word="now!", start=1.8, end=2.5),
        ]
        segment = TranscriptSegment(
            id=0,
            text="Stop scrolling now!",
            start=0.0,
            end=2.5,
            words=words,
        )

        ass_file = tmp_path / "subtitles.ass"
        generator.write_ass_file(ass_file, [segment], clip_start=0.0, clip_end=3.0)

        assert ass_file.is_file()
        content = ass_file.read_text(encoding="utf-8")

        # Verify karaoke animation tags: active word turns yellow (&H0000FFFF&) and bounces (\fscx110\fscy110)
        assert r"{\c&H0000FFFF&\fscx110\fscy110}Stop{\r}" in content
        assert r"{\c&H0000FFFF&\fscx110\fscy110}scrolling{\r}" in content
        assert r"{\c&H00FFFFFF&\fscx100\fscy100}" in content


class TestFFmpegCompositor:
    """Validate async rendering of cropped 9:16 portrait video with burned-in subtitles."""

    def test_alias_compatibility(self) -> None:
        assert VideoCompositor is FFmpegCompositor

    def test_build_filtergraph(self, tmp_path: Path) -> None:
        compositor = FFmpegCompositor()
        dummy_ass = tmp_path / "test.ass"
        dummy_ass.write_text("dummy")

        vf = compositor.build_filtergraph(
            crop_x=656,
            crop_y=0,
            crop_width=608,
            crop_height=1080,
            target_width=1080,
            target_height=1920,
            target_fps=30,
            subtitle_ass_path=dummy_ass,
        )

        assert "crop=608:1080:656:0" in vf
        assert "scale=1080:1920" in vf
        assert "fps=30" in vf
        assert "subtitles=" in vf

    def test_integration_render_dummy_video(
        self,
        dummy_color_bars_video: Path,
        tmp_path: Path,
    ) -> None:
        """Render a 5s 9:16 vertical MP4 with burned-in animated subtitles and probe with ffprobe."""
        import asyncio
        settings = Settings()
        compositor = FFmpegCompositor(settings)
        subtitle_gen = SubtitleGenerator(settings)

        # 1. Generate sample animated subtitle file
        words = [
            WordTimestamp(word="AI", start=0.5, end=1.5),
            WordTimestamp(word="Video", start=1.5, end=2.8),
            WordTimestamp(word="Clipper", start=2.8, end=4.5),
        ]
        segment = TranscriptSegment(
            id=0,
            text="AI Video Clipper",
            start=0.5,
            end=4.5,
            words=words,
        )
        ass_path = tmp_path / "animated.ass"
        subtitle_gen.write_ass_file(ass_path, [segment], clip_start=0.0, clip_end=5.0)

        # 2. Render output portrait MP4
        output_mp4 = tmp_path / "rendered_portrait_clip.mp4"
        crop_w, crop_h = 608, 1080
        crop_x = (1920 - crop_w) // 2

        result = asyncio.run(
            compositor.render_video(
                input_video_path=dummy_color_bars_video,
                output_path=output_mp4,
                start_time=0.0,
                end_time=5.0,
                crop_x=crop_x,
                crop_y=0,
                crop_width=crop_w,
                crop_height=crop_h,
                subtitle_ass_path=ass_path,
                target_width=1080,
                target_height=1920,
                target_fps=30,
                clip_id="test_clip_01",
            )
        )

        assert result.output_path == output_mp4.resolve()
        assert output_mp4.is_file()
        assert output_mp4.stat().st_size > 0

        # 3. Probe the output file using ffprobe to verify valid 9:16 portrait MP4
        ffprobe_bin = str(settings.ffprobe_path)
        probe_cmd = [
            ffprobe_bin,
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,codec_name,pix_fmt,duration",
            "-of", "json",
            str(output_mp4),
        ]

        probe_res = subprocess.run(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        assert probe_res.returncode == 0, f"ffprobe failed: {probe_res.stderr}"

        info = json.loads(probe_res.stdout)
        streams = info.get("streams", [])
        assert len(streams) > 0, "No video stream found in rendered output"

        video_stream = streams[0]
        width = int(video_stream["width"])
        height = int(video_stream["height"])
        codec = video_stream["codec_name"]
        pix_fmt = video_stream["pix_fmt"]

        # Invariants: Strict 9:16 portrait resolution
        assert width == 1080
        assert height == 1920
        assert (width / height) == (9 / 16)
        assert codec == "h264"
        assert pix_fmt == "yuv420p"
