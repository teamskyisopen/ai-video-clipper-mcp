"""Domain entities and Pydantic v2 schemas for AI Video Clipper.

All schemas are strictly validated and decoupled from external transport protocols.
"""

from enum import Enum
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict


class VideoMetadata(BaseModel):
    """Metadata describing a downloaded or locally ingested video."""
    model_config = ConfigDict(frozen=True)

    title: str = Field(description="Title of the video")
    source_url: Optional[str] = Field(default=None, description="Original source URL")
    file_path: Path = Field(description="Local path to the video file")
    duration_seconds: float = Field(ge=0.0, description="Video duration in seconds")
    width: int = Field(gt=0, description="Source video width in pixels")
    height: int = Field(gt=0, description="Source video height in pixels")
    fps: float = Field(gt=0.0, description="Source video frames per second")
    format_note: Optional[str] = Field(default=None, description="Format or codec summary")


class WordTimestamp(BaseModel):
    """Individual word with timestamp and optional confidence score."""
    model_config = ConfigDict(frozen=True)

    word: str = Field(description="Word text")
    start: float = Field(ge=0.0, description="Start time in seconds")
    end: float = Field(ge=0.0, description="End time in seconds")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Confidence score")


# Compatibility alias
TranscriptWord = WordTimestamp


class TranscriptSegment(BaseModel):
    """Segment or sentence block within an audio transcript."""
    model_config = ConfigDict(frozen=True)

    id: int = Field(description="Segment sequence index")
    text: str = Field(description="Full text content of the segment")
    start: float = Field(ge=0.0, description="Segment start timestamp in seconds")
    end: float = Field(ge=0.0, description="Segment end timestamp in seconds")
    words: List[WordTimestamp] = Field(default_factory=list, description="Word-level timestamps")


class TranscriptionResult(BaseModel):
    """Complete structured transcription result."""
    model_config = ConfigDict(frozen=True)

    video_id: str = Field(description="Identifier or name of transcribed media")
    duration: float = Field(ge=0.0, description="Total transcribed audio duration in seconds")
    segments: List[TranscriptSegment] = Field(default_factory=list, description="Ordered transcript segments")
    language: Optional[str] = Field(default="en", description="Detected or specified spoken language")
    full_text: Optional[str] = Field(default="", description="Complete transcript plain text")

    @property
    def duration_seconds(self) -> float:
        """Backwards compatibility duration property."""
        return self.duration


# Compatibility alias
Transcript = TranscriptionResult


class ViralHighlight(BaseModel):
    """Viral highlight moment detected by LLM analysis."""
    model_config = ConfigDict(frozen=True)

    hook_title: str = Field(description="Catchy title or hook headline for the highlight")
    start_time: float = Field(ge=0.0, description="Start timestamp in seconds")
    end_time: float = Field(ge=0.0, description="End timestamp in seconds")
    virality_score: int = Field(ge=0, le=100, description="Virality score (0-100)")
    reasoning: str = Field(description="Rubric-based explanation for virality")
    clean_quote: str = Field(description="Exact transcript quote or hook line")

    @property
    def duration(self) -> float:
        """Calculate duration of the highlight."""
        return max(0.0, self.end_time - self.start_time)


class AnalysisResult(BaseModel):
    """Structured collection of detected viral highlights."""
    model_config = ConfigDict(frozen=True)

    highlights: List[ViralHighlight] = Field(default_factory=list, description="Ranked viral highlights")


class HighlightCandidate(BaseModel):
    """Potential viral clip candidate identified by analysis."""
    model_config = ConfigDict(frozen=True)

    clip_id: str = Field(description="Unique clip identifier")
    start_time: float = Field(ge=0.0, description="Clip start timestamp in seconds")
    end_time: float = Field(ge=0.0, description="Clip end timestamp in seconds")
    title: str = Field(description="Suggested catchy title for the clip")
    hook: str = Field(description="First 3-5 second hook summary or transcript excerpt")
    summary: str = Field(description="Brief summary of the clip content")
    viral_score: float = Field(ge=0.0, le=100.0, description="Virality / retention score (0-100)")
    rationale: str = Field(description="LLM explanation for why this clip was selected")

    @property
    def duration(self) -> float:
        """Calculate duration of candidate clip."""
        return max(0.0, self.end_time - self.start_time)


class FramingCoordinate(BaseModel):
    """Framing crop coordinates and speaker count at a specific timestamp."""
    model_config = ConfigDict(frozen=True)

    timestamp: float = Field(ge=0.0, description="Timestamp in seconds")
    crop_x: int = Field(ge=0, description="Top-left X pixel coordinate")
    crop_y: int = Field(ge=0, description="Top-left Y pixel coordinate")
    width: int = Field(gt=0, description="Crop width in pixels")
    height: int = Field(gt=0, description="Crop height in pixels")
    speaker_count: int = Field(ge=0, description="Detected speaker / face count")
    is_split: bool = Field(default=False, description="Whether this frame represents a dual-speaker split layout")


class FramingBox(BaseModel):
    """Bounding box for detected faces or salient objects."""
    model_config = ConfigDict(frozen=True)

    x: float = Field(ge=0.0, description="Top-left X coordinate normalized (0.0 to 1.0)")
    y: float = Field(ge=0.0, description="Top-left Y coordinate normalized (0.0 to 1.0)")
    width: float = Field(gt=0.0, le=1.0, description="Normalized width (0.0 to 1.0)")
    height: float = Field(gt=0.0, le=1.0, description="Normalized height (0.0 to 1.0)")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Detection confidence")


class CropWindow(BaseModel):
    """Target crop rectangle in absolute pixel coordinates."""
    model_config = ConfigDict(frozen=True)

    x: int = Field(ge=0, description="Top-left X pixel coordinate")
    y: int = Field(ge=0, description="Top-left Y pixel coordinate")
    width: int = Field(gt=0, description="Crop width in pixels")
    height: int = Field(gt=0, description="Crop height in pixels")


class FramingKeyframe(BaseModel):
    """Framing state at a specific video timestamp."""
    model_config = ConfigDict(frozen=True)

    timestamp: float = Field(ge=0.0, description="Keyframe timestamp in seconds")
    active_speaker: bool = Field(default=False, description="Whether active speaker was detected")
    crop_window: CropWindow = Field(description="Target crop window for this moment")


class SubtitleStyle(BaseModel):
    """Styling preferences for burned-in captions."""
    font_name: str = Field(default="Arial", description="Font face")
    font_size: int = Field(default=24, ge=8, description="Font point size")
    primary_color: str = Field(default="&H00FFFFFF", description="ASS color format")
    outline_color: str = Field(default="&H00000000", description="ASS outline color")
    bold: bool = Field(default=True, description="Bold text")
    alignment: int = Field(default=2, description="ASS alignment (2 = bottom-center)")


class RenderPlan(BaseModel):
    """Instructions for video compositor rendering pipeline."""
    model_config = ConfigDict(frozen=True)

    clip_id: str = Field(description="Unique clip identifier")
    source_video_path: Path = Field(description="Path to input source video")
    output_path: Path = Field(description="Destination path for rendered clip")
    start_time: float = Field(ge=0.0, description="Start timestamp in seconds")
    end_time: float = Field(ge=0.0, description="End timestamp in seconds")
    target_width: int = Field(default=1080, ge=100, description="Output width")
    target_height: int = Field(default=1920, ge=100, description="Output height")
    target_fps: int = Field(default=30, ge=1, description="Output FPS")
    keyframes: List[FramingKeyframe] = Field(default_factory=list, description="Dynamic framing crop track")
    burn_subtitles: bool = Field(default=True, description="Whether to burn animated subtitles")
    subtitle_style: SubtitleStyle = Field(default_factory=SubtitleStyle, description="Subtitle visual style")


class JobStatus(str, Enum):
    """Execution status for asynchronous processing jobs."""
    PENDING = "PENDING"
    DOWNLOADING = "DOWNLOADING"
    TRANSCRIBING = "TRANSCRIBING"
    ANALYZING = "ANALYZING"
    FRAMING = "FRAMING"
    RENDERING = "RENDERING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ClipResult(BaseModel):
    """Outcome of a successfully rendered clip."""
    model_config = ConfigDict(frozen=True)

    clip_id: str = Field(description="Unique clip identifier")
    output_path: Path = Field(description="Absolute path to generated short video file")
    duration_seconds: float = Field(ge=0.0, description="Final rendered clip duration")
    title: str = Field(description="Clip title")
    viral_score: float = Field(ge=0.0, le=100.0, description="Calculated virality score")
    file_size_bytes: int = Field(ge=0, description="File size in bytes")
