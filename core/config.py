"""Central application configuration using Pydantic Settings.

Enforces ZERO hardcoding rule across the entire application.
"""

from functools import lru_cache
from pathlib import Path
import shutil
import tempfile
from typing import Optional, Generator
from contextlib import contextmanager

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Immutable application settings loaded from environment or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application & Logging
    app_name: str = Field(default="ai-video-clipper-mcp", description="Application service name")
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: str = Field(default="INFO", description="Structured logging level")

    # Paths & Binaries
    bin_dir: Path = Field(default=Path("./bin"), description="Local vendor binaries directory")
    ffmpeg_path: Path = Field(default=Path("./bin/ffmpeg.exe"), description="Path to ffmpeg binary")
    ffprobe_path: Path = Field(default=Path("./bin/ffprobe.exe"), description="Path to ffprobe binary")
    temp_storage_dir: Path = Field(
        default=Path("./storage/temp"),
        description="Root temporary storage path for media processing",
    )
    output_dir: Path = Field(
        default=Path("./storage/output"),
        description="Target directory for rendered short clips",
    )

    # Video Cropping & Target Dimensions (Default 9:16 portrait)
    target_width: int = Field(default=1080, ge=100, description="Target clip width in pixels")
    target_height: int = Field(default=1920, ge=100, description="Target clip height in pixels")
    target_fps: int = Field(default=30, ge=1, le=120, description="Target output frame rate")
    min_clip_duration_seconds: int = Field(default=15, ge=1, description="Minimum clip duration")
    max_clip_duration_seconds: int = Field(default=60, ge=5, description="Maximum clip duration")

    # Framing, Scene Detection & Smoothing
    framer_smoothing_alpha: float = Field(default=0.25, ge=0.01, le=1.0, description="EMA smoothing alpha for crop tracking")
    scene_detection_threshold: float = Field(default=27.0, ge=1.0, description="Scene detection sensitivity threshold")
    min_scene_duration_seconds: float = Field(default=0.8, ge=0.1, description="Minimum scene duration in seconds")
    split_screen_enabled: bool = Field(default=True, description="Enable dual-speaker split screen mode")
    split_min_separation: float = Field(default=0.20, ge=0.05, le=0.8, description="Min horizontal distance ratio for split")

    # Transcription Engine
    whisper_model: str = Field(default="base", description="Whisper model size/name")
    whisper_device: str = Field(default="auto", description="Execution device for transcription (cpu/cuda/auto)")
    whisper_compute_type: str = Field(default="default", description="Quantization/compute type")

    # LLM / Analysis Engine
    llm_provider: str = Field(default="gemini", description="LLM provider name ('gemini', 'openai', 'none')")
    llm_model: str = Field(default="gemini-3.8-flash", description="LLM model identifier")
    llm_api_key: Optional[str] = Field(default=None, description="Generic API Key for LLM provider")
    llm_base_url: Optional[str] = Field(default=None, description="Custom base URL for LLM provider")
    gemini_api_key: Optional[str] = Field(default=None, description="Google Gemini API Key")
    gemini_model: str = Field(default="gemini-3.8-flash", description="Gemini model name")
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API Key")
    openai_model: str = Field(default="gpt-4o", description="OpenAI model name")
    openai_base_url: Optional[str] = Field(default=None, description="Custom base URL for OpenAI-compatible API")
    offline_mode: bool = Field(default=False, description="Disable external LLM calls and use offline heuristic analysis")

    # Subtitle Styling & Positioning
    subtitle_font_name: str = Field(default="Arial", description="Subtitle font family")
    subtitle_font_size: int = Field(default=22, ge=8, description="Subtitle font point size")
    subtitle_primary_color: str = Field(default="&H00FFFFFF&", description="Default inactive subtitle text color")
    subtitle_highlight_color: str = Field(default="&H0000FFFF&", description="Active spoken karaoke highlight color (yellow)")
    subtitle_outline_color: str = Field(default="&H00000000&", description="Subtitle outline color (black)")
    subtitle_margin_v: int = Field(default=250, ge=10, description="Bottom safe margin avoiding TikTok/Shorts UI")
    crf: int = Field(default=22, ge=0, le=51, description="Constant rate factor for x264")
    ffmpeg_preset: str = Field(default="fast", description="FFmpeg x264 preset")

    # Network & Ingestion Settings
    proxy: Optional[str] = Field(default=None, description="HTTP/HTTPS/SOCKS proxy for network calls")
    download_rate_limit: Optional[str] = Field(default=None, description="Download speed limit (e.g., '50K', '1M')")
    audio_codec: str = Field(default="mp3", description="Audio extraction codec ('mp3', 'aac', etc.)")
    audio_bitrate: str = Field(default="128k", description="Audio extraction bitrate ('128k', '192k', etc.)")

    # Timeouts (in seconds)
    download_timeout_seconds: int = Field(default=600, ge=10, description="Max allowed download duration")
    transcription_timeout_seconds: int = Field(default=600, ge=10, description="Max allowed transcription duration")
    analysis_timeout_seconds: int = Field(default=180, ge=10, description="Max allowed LLM analysis duration")
    render_timeout_seconds: int = Field(default=600, ge=10, description="Max allowed FFmpeg render duration")

    @field_validator("ffmpeg_path", "ffprobe_path", mode="after")
    @classmethod
    def resolve_binary(cls, binary_path: Path) -> Path:
        """Resolve binary from local bin directory, or locate in system PATH if missing."""
        if binary_path.is_file():
            return binary_path.resolve()
        
        # Check both extensionless (Linux/macOS) and .exe (Windows) in bin directory
        bin_dir = binary_path.parent
        stem = binary_path.stem
        for candidate_name in [f"{stem}.exe", stem]:
            cand = bin_dir / candidate_name
            if cand.is_file():
                return cand.resolve()

        # Fall back to system executable lookup if local file doesn't exist
        for candidate_name in [binary_path.name, stem]:
            found = shutil.which(candidate_name)
            if found:
                return Path(found).resolve()

        return binary_path.resolve()

    def ensure_directories(self) -> None:
        """Ensure non-ephemeral directories exist on filesystem."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_storage_dir.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def create_managed_temp_dir(self, prefix: str = "clipper_") -> Generator[Path, None, None]:
        """Managed auto-cleaning temporary directory context within the configured temp directory."""
        self.ensure_directories()
        temp_dir = Path(tempfile.mkdtemp(prefix=prefix, dir=self.temp_storage_dir))
        try:
            yield temp_dir
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton provider for application settings."""
    settings = Settings()
    return settings
