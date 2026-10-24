"""Core module exports: settings, exceptions, and core utilities."""

from core.config import Settings, get_settings
from core.exceptions import (
    VideoClipperError,
    ConfigurationError,
    MediaDownloadError,
    TranscriptionError,
    HighlightAnalysisError,
    FramingError,
    CompositionError,
    StorageError,
)

__all__ = [
    "Settings",
    "get_settings",
    "VideoClipperError",
    "ConfigurationError",
    "MediaDownloadError",
    "TranscriptionError",
    "HighlightAnalysisError",
    "FramingError",
    "CompositionError",
    "StorageError",
]
