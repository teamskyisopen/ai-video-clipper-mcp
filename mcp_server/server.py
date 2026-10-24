"""FastMCP Server exposing AI Video Clipper tools.

Driving adapter orchestrating services via PipelineCoordinator and FastMCP protocol.
"""

import asyncio
from typing import Any, Dict, Optional
from loguru import logger

try:
    from mcp.server.fastmcp import FastMCP
except (ImportError, ModuleNotFoundError):
    from mcp.server.mcpserver import MCPServer as FastMCP

from core.config import get_settings
from services.coordinator import PipelineCoordinator

# Initialize settings and pipeline coordinator
settings = get_settings()
coordinator = PipelineCoordinator(settings)

# Initialize MCP application
mcp = FastMCP(settings.app_name)


@mcp.tool()
async def analyze_video_highlights(url: str, max_clips: int = 3) -> Dict[str, Any]:
    """Download audio, transcribe, analyze viral highlights, and return ranked moments with timestamps.

    Args:
        url: Video URL to analyze (YouTube or direct media URL).
        max_clips: Maximum number of viral highlight clips to detect (default: 3).
    """
    logger.info("MCP Tool: analyze_video_highlights invoked", url=url, max_clips=max_clips)
    try:
        analysis = await coordinator.analyze_video_highlights(url=url, max_clips=max_clips)
        return {
            "status": "success",
            "url": url,
            "highlights_count": len(analysis.highlights),
            "highlights": [
                {
                    "hook_title": h.hook_title,
                    "start_time": h.start_time,
                    "end_time": h.end_time,
                    "duration_seconds": h.duration,
                    "virality_score": h.virality_score,
                    "reasoning": h.reasoning,
                    "clean_quote": h.clean_quote,
                }
                for h in analysis.highlights
            ],
        }
    except Exception as exc:
        logger.error("analyze_video_highlights tool error", error=str(exc))
        return {
            "status": "error",
            "message": str(exc),
        }


@mcp.tool()
async def render_highlight_clip(
    url: str,
    start_time: float,
    end_time: float,
    subtitle_preset: str = "karaoke_yellow",
) -> Dict[str, Any]:
    """Download video slice, calculate portrait 9:16 crop, burn animated subtitles, and render MP4 clip.

    Args:
        url: Video source URL.
        start_time: Start timestamp in seconds.
        end_time: End timestamp in seconds.
        subtitle_preset: Visual styling for subtitles (default: 'karaoke_yellow').
    """
    logger.info(
        "MCP Tool: render_highlight_clip invoked",
        url=url,
        start=start_time,
        end=end_time,
        preset=subtitle_preset,
    )
    try:
        result = await coordinator.render_highlight_clip(
            url=url,
            start_time=start_time,
            end_time=end_time,
            subtitle_preset=subtitle_preset,
        )
        return {
            "status": "success",
            "clip_id": result.clip_id,
            "output_path": str(result.output_path),
            "duration_seconds": result.duration_seconds,
            "file_size_bytes": result.file_size_bytes,
        }
    except Exception as exc:
        logger.error("render_highlight_clip tool error", error=str(exc))
        return {
            "status": "error",
            "message": str(exc),
        }


@mcp.tool()
async def auto_clip_video(url: str, clip_count: int = 3) -> Dict[str, Any]:
    """End-to-end macro tool: downloads, analyzes, crops, subtitles, and renders all top viral clips.

    Args:
        url: Source video URL.
        clip_count: Number of viral clips to detect and render (default: 3).
    """
    logger.info("MCP Tool: auto_clip_video invoked", url=url, clip_count=clip_count)
    try:
        results = await coordinator.auto_clip_video(url=url, clip_count=clip_count)
        return {
            "status": "success",
            "rendered_count": len(results),
            "clips": [
                {
                    "clip_id": r.clip_id,
                    "output_path": str(r.output_path),
                    "duration_seconds": r.duration_seconds,
                    "file_size_bytes": r.file_size_bytes,
                }
                for r in results
            ],
        }
    except Exception as exc:
        logger.error("auto_clip_video tool error", error=str(exc))
        return {
            "status": "error",
            "message": str(exc),
        }


def main() -> None:
    """Run MCP server over stdio transport."""
    logger.info("Starting AI Video Clipper MCP server on stdio...")
    mcp.run()


if __name__ == "__main__":
    main()
