"""Media downloader service using yt-dlp.

Handles video, audio extraction, and time-range video slicing with dynamic configuration.
"""

from pathlib import Path
from typing import Optional
from loguru import logger

from core.config import Settings, get_settings
from core.exceptions import MediaDownloadError
from domain.models import VideoMetadata


class MediaDownloader:
    """Downloader port/adapter wrapping yt-dlp media retrieval."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()

    def _build_base_ydl_opts(self) -> dict:
        """Construct common yt-dlp options based on active settings."""
        opts = {
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "extractor_args": {"youtube": {"player_client": ["android", "web"]}},
        }

        # Apply proxy if configured
        if self.settings.proxy:
            opts["proxy"] = self.settings.proxy

        # Apply download rate limit if configured
        if self.settings.download_rate_limit:
            try:
                import yt_dlp.utils
                opts["ratelimit"] = yt_dlp.utils.parse_bytes(self.settings.download_rate_limit)
            except Exception as exc:
                logger.warning(
                    "Failed to parse download_rate_limit",
                    limit=self.settings.download_rate_limit,
                    error=str(exc),
                )

        # Configure local FFmpeg directory and inject into PATH for yt-dlp partial downloaders
        if self.settings.ffmpeg_path.exists():
            bin_dir = str(self.settings.ffmpeg_path.parent.resolve())
            import os
            path_env = os.environ.get("PATH", "")
            if bin_dir not in path_env:
                os.environ["PATH"] = bin_dir + os.pathsep + path_env
            opts["ffmpeg_location"] = bin_dir

        return opts

    def fetch_metadata(self, source_url: str) -> dict:
        """Extract video metadata without downloading full stream."""
        logger.info("Fetching video metadata", url=source_url)
        try:
            import yt_dlp
            ydl_opts = self._build_base_ydl_opts()
            ydl_opts.update({
                "extract_flat": False,
                "skip_download": True,
            })
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(source_url, download=False)
                if not info:
                    raise MediaDownloadError(f"No metadata returned for URL: {source_url}")
                return info
        except Exception as exc:
            logger.error("Failed to extract metadata", url=source_url, error=str(exc))
            raise MediaDownloadError(f"Failed to fetch metadata: {exc}") from exc

    def download_audio(self, url: str, destination_dir: Optional[Path] = None) -> Path:
        """Download and extract fast 128kbps audio track (MP3/AAC)."""
        dest_dir = destination_dir or self.settings.temp_storage_dir
        dest_dir.mkdir(parents=True, exist_ok=True)
        out_template = str(dest_dir / "%(id)s.%(ext)s")

        codec = self.settings.audio_codec.lower().strip()
        bitrate = self.settings.audio_bitrate.lower().replace("k", "").strip()

        logger.info(
            "Starting fast audio download",
            url=url,
            codec=codec,
            bitrate=bitrate,
            destination=str(dest_dir),
        )

        try:
            import yt_dlp
            ydl_opts = self._build_base_ydl_opts()
            ydl_opts.update({
                "outtmpl": out_template,
                "format": "ba/b",
                "postprocessors": [
                    {
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": codec,
                        "preferredquality": bitrate,
                    }
                ],
            })

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                raw_filename = Path(ydl.prepare_filename(info))
                audio_path = raw_filename.with_suffix(f".{codec}")

                if not audio_path.exists():
                    # Check for alternative audio formats produced by extractor
                    candidates = list(dest_dir.glob(f"{raw_filename.stem}*"))
                    for cand in candidates:
                        if cand.suffix.lower() in [".mp3", ".m4a", ".aac", ".opus", ".wav"]:
                            audio_path = cand
                            break

                logger.info("Audio extracted successfully", path=str(audio_path))
                return audio_path.resolve()

        except Exception as exc:
            logger.error("Audio download failed", url=url, error=str(exc))
            raise MediaDownloadError(f"Failed to download audio for {url}: {exc}") from exc

    def download_video_slice(
        self,
        url: str,
        start: float,
        end: float,
        destination_dir: Optional[Path] = None,
    ) -> Path:
        """Download only a specific time range of a video using download sections."""
        dest_dir = destination_dir or self.settings.temp_storage_dir
        dest_dir.mkdir(parents=True, exist_ok=True)
        out_template = str(dest_dir / f"%(id)s_slice_{int(start)}_{int(end)}.%(ext)s")

        logger.info(
            "Downloading video slice",
            url=url,
            start=start,
            end=end,
            destination=str(dest_dir),
        )

        try:
            import yt_dlp
            import yt_dlp.utils

            ydl_opts = self._build_base_ydl_opts()
            ydl_opts.update({
                "outtmpl": out_template,
                "format": "bestvideo+bestaudio/best",
                "merge_output_format": "mp4",
                "download_ranges": yt_dlp.utils.download_range_func(None, [(start, end)]),
                "force_keyframes_at_cuts": True,
            })

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                filename = Path(ydl.prepare_filename(info))

                if not filename.exists():
                    mp4_cand = filename.with_suffix(".mp4")
                    if mp4_cand.exists():
                        filename = mp4_cand
                    else:
                        matches = list(dest_dir.glob(f"{filename.stem}*"))
                        if matches:
                            filename = matches[0]

                logger.info("Video slice downloaded successfully", path=str(filename))
                return filename.resolve()

        except Exception as exc:
            logger.error("Video slice download failed", url=url, start=start, end=end, error=str(exc))
            raise MediaDownloadError(f"Failed to download slice for {url}: {exc}") from exc

    def download(self, source_url: str, destination_dir: Optional[Path] = None) -> VideoMetadata:
        """Download full video into destination directory."""
        dest_dir = destination_dir or self.settings.temp_storage_dir
        dest_dir.mkdir(parents=True, exist_ok=True)
        out_template = str(dest_dir / "%(id)s.%(ext)s")

        logger.info("Starting full video download", url=source_url, destination=str(dest_dir))
        try:
            import yt_dlp
            ydl_opts = self._build_base_ydl_opts()
            ydl_opts.update({
                "outtmpl": out_template,
                "format": "bestvideo+bestaudio/best",
                "merge_output_format": "mp4",
            })
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(source_url, download=True)
                downloaded_file = Path(ydl.prepare_filename(info))

                if not downloaded_file.exists():
                    merged_candidate = downloaded_file.with_suffix(".mp4")
                    if merged_candidate.exists():
                        downloaded_file = merged_candidate

                metadata = VideoMetadata(
                    title=info.get("title", "Untitled Video"),
                    source_url=source_url,
                    file_path=downloaded_file.resolve(),
                    duration_seconds=float(info.get("duration", 0.0)),
                    width=int(info.get("width") or 1920),
                    height=int(info.get("height") or 1080),
                    fps=float(info.get("fps") or 30.0),
                    format_note=info.get("format_note"),
                )
                logger.info(
                    "Video download completed successfully",
                    path=str(downloaded_file),
                    duration=metadata.duration_seconds,
                )
                return metadata
        except Exception as exc:
            logger.error("Download failed", url=source_url, error=str(exc))
            raise MediaDownloadError(f"Download failed for {source_url}: {exc}") from exc


# Backwards compatibility alias
VideoDownloader = MediaDownloader
