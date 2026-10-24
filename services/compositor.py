"""Video compositor service using FFmpeg.

Constructs filtergraphs for cropping, portrait scaling, and ASS subtitle burn-in.
Executes rendering asynchronously via asyncio.create_subprocess_exec.
"""

import asyncio
from pathlib import Path
from typing import Optional
from loguru import logger

from core.config import Settings, get_settings
from core.exceptions import CompositionError
from domain.models import RenderPlan, ClipResult


class FFmpegCompositor:
    """Video compositing engine driving FFmpeg with async process execution."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()

    def build_filtergraph(
        self,
        crop_x: Optional[int] = None,
        crop_y: Optional[int] = None,
        crop_width: Optional[int] = None,
        crop_height: Optional[int] = None,
        target_width: Optional[int] = None,
        target_height: Optional[int] = None,
        target_fps: Optional[int] = None,
        subtitle_ass_path: Optional[Path] = None,
    ) -> str:
        """Assemble clean FFmpeg -vf filter chain string."""
        out_w = target_width or self.settings.target_width
        out_h = target_height or self.settings.target_height
        out_fps = target_fps or self.settings.target_fps

        filters = []

        # 1. Spatial crop
        if crop_width is not None and crop_height is not None:
            cx = max(0, crop_x or 0)
            cy = max(0, crop_y or 0)
            filters.append(f"crop={crop_width}:{crop_height}:{cx}:{cy}")

        # 2. Scale & FPS
        filters.append(f"scale={out_w}:{out_h}")
        filters.append(f"fps={out_fps}")

        # 3. Subtitle Burn-In
        if subtitle_ass_path is not None and subtitle_ass_path.is_file():
            # Escape path for FFmpeg filter syntax (especially Windows colons and slashes)
            escaped_path = str(subtitle_ass_path.resolve()).replace("\\", "/").replace(":", "\\:")
            filters.append(f"subtitles='{escaped_path}'")

        return ",".join(filters)

    async def render_video(
        self,
        input_video_path: Path,
        output_path: Path,
        start_time: float,
        end_time: float,
        crop_x: Optional[int] = None,
        crop_y: Optional[int] = None,
        crop_width: Optional[int] = None,
        crop_height: Optional[int] = None,
        subtitle_ass_path: Optional[Path] = None,
        target_width: Optional[int] = None,
        target_height: Optional[int] = None,
        target_fps: Optional[int] = None,
        clip_id: str = "clip_001",
    ) -> ClipResult:
        """Asynchronously render video slice with cropping and subtitles."""
        if not input_video_path.is_file():
            raise CompositionError(f"Input video file not found: {input_video_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg_bin = str(self.settings.ffmpeg_path)
        duration = max(0.1, end_time - start_time)

        vf_arg = self.build_filtergraph(
            crop_x=crop_x,
            crop_y=crop_y,
            crop_width=crop_width,
            crop_height=crop_height,
            target_width=target_width,
            target_height=target_height,
            target_fps=target_fps,
            subtitle_ass_path=subtitle_ass_path,
        )

        cmd = [
            ffmpeg_bin,
            "-y",
            "-ss", str(start_time),
            "-i", str(input_video_path),
            "-t", str(duration),
            "-vf", vf_arg,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", self.settings.ffmpeg_preset,
            "-crf", str(self.settings.crf),
            "-c:a", "aac",
            "-b:a", "192k",
            "-movflags", "+faststart",
            str(output_path),
        ]

        logger.info(
            "Spawning FFmpeg async compositing process",
            clip_id=clip_id,
            input=str(input_video_path),
            output=str(output_path),
            duration=duration,
        )
        logger.debug("FFmpeg filtergraph", vf=vf_arg)

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=float(self.settings.render_timeout_seconds),
            )

            if process.returncode != 0:
                err_text = stderr.decode(errors="replace")
                logger.error("FFmpeg execution error", returncode=process.returncode, stderr=err_text)
                raise CompositionError(
                    f"FFmpeg failed with exit code {process.returncode}: {err_text[-500:]}"
                )

            if not output_path.exists() or output_path.stat().st_size == 0:
                raise CompositionError(f"Rendered output file was not created: {output_path}")

            file_size = output_path.stat().st_size
            logger.info("Video rendered successfully", path=str(output_path), size_bytes=file_size)

            return ClipResult(
                clip_id=clip_id,
                output_path=output_path.resolve(),
                duration_seconds=duration,
                title=f"Rendered Clip {clip_id}",
                viral_score=90.0,
                file_size_bytes=file_size,
            )

        except asyncio.TimeoutError as exc:
            raise CompositionError("FFmpeg process timed out") from exc
        except Exception as exc:
            if isinstance(exc, CompositionError):
                raise
            logger.error("Unexpected error during video compositing", error=str(exc))
            raise CompositionError(f"Compositing error: {exc}") from exc

    def render_clip(self, plan: RenderPlan) -> ClipResult:
        """Synchronous wrapper executing render_video for compatibility."""
        crop_win = plan.keyframes[0].crop_window if plan.keyframes else None

        crop_x = crop_win.x if crop_win else None
        crop_y = crop_win.y if crop_win else None
        crop_w = crop_win.width if crop_win else None
        crop_h = crop_win.height if crop_win else None

        # Execute async render in event loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        coro = self.render_video(
            input_video_path=plan.source_video_path,
            output_path=plan.output_path,
            start_time=plan.start_time,
            end_time=plan.end_time,
            crop_x=crop_x,
            crop_y=crop_y,
            crop_width=crop_w,
            crop_height=crop_h,
            target_width=plan.target_width,
            target_height=plan.target_height,
            target_fps=plan.target_fps,
            clip_id=plan.clip_id,
        )

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result()
        else:
            return asyncio.run(coro)


# Backwards compatibility alias
VideoCompositor = FFmpegCompositor
