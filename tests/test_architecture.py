"""Architecture and schema invariant tests."""

from pathlib import Path
import pytest
from pydantic import ValidationError

from core.config import Settings
from core.exceptions import VideoClipperError
from domain.models import (
    HighlightCandidate,
    RenderPlan,
    VideoMetadata,
)


def test_settings_defaults() -> None:
    """Verify that settings load with strict defaults and no hardcoded fallbacks."""
    settings = Settings()
    assert settings.app_name == "ai-video-clipper-mcp"
    assert settings.target_width == 1080
    assert settings.target_height == 1920
    assert settings.target_fps == 30
    assert settings.min_clip_duration_seconds == 15
    assert settings.max_clip_duration_seconds == 60


def test_ffmpeg_binary_detection() -> None:
    """Verify FFmpeg binary locator detects local binaries or system PATH."""
    settings = Settings()
    assert settings.ffmpeg_path is not None
    assert settings.ffprobe_path is not None


def test_domain_model_validation() -> None:
    """Verify that Pydantic v2 domain schemas enforce strict types and reject invalid payloads."""
    # Valid model
    candidate = HighlightCandidate(
        clip_id="test_01",
        start_time=10.0,
        end_time=35.0,
        title="Test Title",
        hook="Attention grabber",
        summary="Summary text",
        viral_score=92.5,
        rationale="Strong emotional hook",
    )
    assert candidate.duration == 25.0

    # Invalid viral score > 100
    with pytest.raises(ValidationError):
        HighlightCandidate(
            clip_id="test_02",
            start_time=0.0,
            end_time=10.0,
            title="Invalid Score",
            hook="Hook",
            summary="Summary",
            viral_score=150.0,
            rationale="Test",
        )


def test_managed_temp_lifecycle(tmp_path: Path) -> None:
    """Verify that managed temporary directories cleanly auto-delete."""
    settings = Settings(temp_storage_dir=tmp_path)
    created_path = None
    with settings.create_managed_temp_dir() as temp_dir:
        created_path = temp_dir
        assert temp_dir.exists()
        test_file = temp_dir / "sample.txt"
        test_file.write_text("temporary data")
        assert test_file.exists()

    # Outside context manager, directory should be removed
    assert not created_path.exists()
