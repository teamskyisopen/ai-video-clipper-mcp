"""Custom domain and infrastructure exceptions for AI Video Clipper MCP."""

from typing import Optional


class VideoClipperError(Exception):
    """Base exception for all domain and service errors in the video clipper."""

    def __init__(self, message: str, details: Optional[dict] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigurationError(VideoClipperError):
    """Raised when environment variables or configuration settings are invalid."""
    pass


class MediaDownloadError(VideoClipperError):
    """Raised when video or audio download fails."""
    pass


class TranscriptionError(VideoClipperError):
    """Raised when extracting audio or generating transcript fails."""
    pass


class HighlightAnalysisError(VideoClipperError):
    """Raised when transcript analysis or LLM clip extraction fails."""
    pass


class FramingError(VideoClipperError):
    """Raised when face detection or crop calculation fails."""
    pass


class CompositionError(VideoClipperError):
    """Raised when FFmpeg rendering, filtering, or compositing fails."""
    pass


class StorageError(VideoClipperError):
    """Raised when creating or cleaning temporary storage directories fails."""
    pass
