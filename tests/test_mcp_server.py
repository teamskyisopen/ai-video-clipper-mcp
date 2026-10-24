"""Unit and integration tests for FastMCP server tools and PipelineCoordinator."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from core.config import Settings
from domain.models import (
    AnalysisResult,
    ClipResult,
    FramingCoordinate,
    TranscriptSegment,
    TranscriptionResult,
    ViralHighlight,
    WordTimestamp,
)
from services.coordinator import PipelineCoordinator
from mcp_server.server import (
    mcp,
    analyze_video_highlights,
    render_highlight_clip,
    auto_clip_video,
)


@pytest.fixture
def mock_analysis_result() -> AnalysisResult:
    return AnalysisResult(
        highlights=[
            ViralHighlight(
                hook_title="Viral Hook 1",
                start_time=10.0,
                end_time=40.0,
                virality_score=95,
                reasoning="Compelling curiosity gap",
                clean_quote="You won't believe this trick.",
            ),
            ViralHighlight(
                hook_title="Viral Hook 2",
                start_time=50.0,
                end_time=80.0,
                virality_score=88,
                reasoning="Strong emotional payoff",
                clean_quote="Here is why it matters.",
            ),
        ]
    )


class TestPipelineCoordinator:
    """Validate PipelineCoordinator service orchestration."""

    def test_coordinator_initialization(self) -> None:
        settings = Settings()
        coordinator = PipelineCoordinator(settings)
        assert coordinator.downloader is not None
        assert coordinator.transcriber is not None
        assert coordinator.analyzer is not None
        assert coordinator.framer is not None
        assert coordinator.subtitles is not None
        assert coordinator.compositor is not None

    def test_analyze_video_highlights_orchestration(
        self,
        mock_analysis_result: AnalysisResult,
        tmp_path: Path,
    ) -> None:
        settings = Settings(temp_storage_dir=tmp_path)

        mock_downloader = MagicMock()
        mock_downloader.download_audio.return_value = tmp_path / "mock_audio.mp3"

        mock_transcriber = MagicMock()
        mock_transcriber.transcribe.return_value = TranscriptionResult(
            video_id="test_vid",
            duration=90.0,
            segments=[],
        )

        mock_analyzer = MagicMock()
        mock_analyzer.detect_highlights.return_value = mock_analysis_result

        coordinator = PipelineCoordinator(
            settings=settings,
            downloader=mock_downloader,
            transcriber=mock_transcriber,
            analyzer=mock_analyzer,
        )

        result = asyncio.run(
            coordinator.analyze_video_highlights("https://youtube.com/watch?v=123", max_clips=2)
        )

        assert isinstance(result, AnalysisResult)
        assert len(result.highlights) == 2
        assert result.highlights[0].hook_title == "Viral Hook 1"
        assert mock_downloader.download_audio.called
        assert mock_transcriber.transcribe.called
        assert mock_analyzer.detect_highlights.called

    def test_render_highlight_clip_orchestration(self, tmp_path: Path) -> None:
        settings = Settings(
            temp_storage_dir=tmp_path,
            output_dir=tmp_path / "output",
        )

        mock_downloader = MagicMock()
        mock_downloader.download_video_slice.return_value = tmp_path / "slice.mp4"

        mock_framer = MagicMock()
        mock_framer.calculate_framing_trajectory.return_value = [
            FramingCoordinate(
                timestamp=0.0,
                crop_x=650,
                crop_y=0,
                width=608,
                height=1080,
                speaker_count=1,
            )
        ]

        mock_subtitles = MagicMock()
        mock_subtitles.write_ass_file.return_value = tmp_path / "subs.ass"

        fake_clip_result = ClipResult(
            clip_id="test_clip",
            output_path=tmp_path / "output" / "test_clip.mp4",
            duration_seconds=30.0,
            title="Clip",
            viral_score=90.0,
            file_size_bytes=1024,
        )

        mock_compositor = MagicMock()
        mock_compositor.render_video = AsyncMock(return_value=fake_clip_result)

        coordinator = PipelineCoordinator(
            settings=settings,
            downloader=mock_downloader,
            framer=mock_framer,
            subtitles=mock_subtitles,
            compositor=mock_compositor,
        )

        result = asyncio.run(
            coordinator.render_highlight_clip(
                url="https://youtube.com/watch?v=123",
                start_time=10.0,
                end_time=40.0,
                clip_id="test_clip",
            )
        )

        assert result.clip_id == "test_clip"
        assert result.duration_seconds == 30.0
        assert mock_downloader.download_video_slice.called
        assert mock_framer.calculate_framing_trajectory.called
        assert mock_compositor.render_video.called


class TestMCPTools:
    """Validate exposed MCP async tool functions."""

    def test_tools_registered_on_server(self) -> None:
        """Verify the 3 required tools are registered in the FastMCP application."""
        tools = asyncio.run(mcp.list_tools())
        tool_names = [t.name for t in tools]
        assert "analyze_video_highlights" in tool_names
        assert "render_highlight_clip" in tool_names
        assert "auto_clip_video" in tool_names

    def test_analyze_video_highlights_tool(
        self,
        mock_analysis_result: AnalysisResult,
    ) -> None:
        with patch(
            "mcp_server.server.coordinator.analyze_video_highlights",
            new=AsyncMock(return_value=mock_analysis_result),
        ):
            res = asyncio.run(
                analyze_video_highlights("https://youtube.com/watch?v=abc", max_clips=2)
            )
            assert res["status"] == "success"
            assert res["highlights_count"] == 2
            assert res["highlights"][0]["hook_title"] == "Viral Hook 1"
            assert res["highlights"][0]["virality_score"] == 95

    def test_render_highlight_clip_tool(self, tmp_path: Path) -> None:
        fake_result = ClipResult(
            clip_id="clip_abc",
            output_path=tmp_path / "clip_abc.mp4",
            duration_seconds=25.0,
            title="Clip ABC",
            viral_score=92.0,
            file_size_bytes=4096,
        )

        with patch(
            "mcp_server.server.coordinator.render_highlight_clip",
            new=AsyncMock(return_value=fake_result),
        ):
            res = asyncio.run(
                render_highlight_clip("https://youtube.com/watch?v=abc", 15.0, 40.0)
            )
            assert res["status"] == "success"
            assert res["clip_id"] == "clip_abc"
            assert res["duration_seconds"] == 25.0
            assert "clip_abc.mp4" in res["output_path"]

    def test_auto_clip_video_tool(self, tmp_path: Path) -> None:
        fake_results = [
            ClipResult(
                clip_id="autoclip_01",
                output_path=tmp_path / "autoclip_01.mp4",
                duration_seconds=30.0,
                title="Clip 1",
                viral_score=94.0,
                file_size_bytes=5000,
            )
        ]

        with patch(
            "mcp_server.server.coordinator.auto_clip_video",
            new=AsyncMock(return_value=fake_results),
        ):
            res = asyncio.run(auto_clip_video("https://youtube.com/watch?v=abc", clip_count=1))
            assert res["status"] == "success"
            assert res["rendered_count"] == 1
            assert res["clips"][0]["clip_id"] == "autoclip_01"
