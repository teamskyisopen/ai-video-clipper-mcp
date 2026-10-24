"""Subtitle generation service producing Advanced SubStation Alpha (.ass) scripts.

Implements word-level karaoke animations and safe margin positioning for short-form video platforms.
"""

from pathlib import Path
from typing import List, Optional
from loguru import logger

from core.config import Settings, get_settings
from domain.models import TranscriptSegment, WordTimestamp


def format_ass_timestamp(seconds: float) -> str:
    """Format floating point seconds into ASS timestamp format: H:MM:SS.cs."""
    clamped = max(0.0, seconds)
    hours = int(clamped // 3600)
    minutes = int((clamped % 3600) // 60)
    secs = int(clamped % 60)
    centis = int(round((clamped - int(clamped)) * 100))
    if centis >= 100:
        secs += 1
        centis -= 100
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


class SubtitleGenerator:
    """Generates styled ASS subtitles with word-level karaoke highlighting."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()

    def generate_ass_header(self, is_split_screen: bool = False) -> str:
        """Construct the ASS script info and V4+ style definition."""
        font_name = self.settings.subtitle_font_name
        # Scale font size appropriately for target video height
        font_size = int(self.settings.subtitle_font_size * (self.settings.target_height / 720.0))
        primary_color = self.settings.subtitle_primary_color
        outline_color = self.settings.subtitle_outline_color

        # Margin and alignment: If split-screen, center vertically (Alignment 5);
        # else place in safe bottom third (Alignment 2) with MarginV
        alignment = 5 if is_split_screen else 2
        margin_v = 0 if is_split_screen else self.settings.subtitle_margin_v

        header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {self.settings.target_width}
PlayResY: {self.settings.target_height}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{font_size},{primary_color},&H000000FF,{outline_color},&H80000000,-1,0,0,0,100,100,0,0,1,4,2,{alignment},40,40,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        return header

    def generate_ass_content(
        self,
        segments: List[TranscriptSegment],
        clip_start: float = 0.0,
        clip_end: Optional[float] = None,
        is_split_screen: bool = False,
    ) -> str:
        """Build full ASS script with word-level karaoke animated dialogue events."""
        lines = [self.generate_ass_header(is_split_screen=is_split_screen)]

        highlight_color = self.settings.subtitle_highlight_color
        primary_color = self.settings.subtitle_primary_color

        for seg in segments:
            # Skip segments outside clip window
            if clip_end is not None and seg.start >= clip_end:
                continue
            if seg.end <= clip_start:
                continue

            words = seg.words
            if not words:
                # If no word-level timestamps, render simple segment line
                rel_start = max(0.0, seg.start - clip_start)
                rel_end = (min(seg.end, clip_end) if clip_end else seg.end) - clip_start
                if rel_end > rel_start:
                    start_str = format_ass_timestamp(rel_start)
                    end_str = format_ass_timestamp(rel_end)
                    text = seg.text.strip().replace("\n", " ")
                    lines.append(f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{text}")
                continue

            # Word-level karaoke highlighting:
            # Emit an event for each word window where the active word turns yellow with bounce
            for active_idx, active_word in enumerate(words):
                w_start = max(0.0, active_word.start - clip_start)
                w_end = (min(active_word.end, clip_end) if clip_end else active_word.end) - clip_start

                if w_end <= w_start:
                    continue

                start_str = format_ass_timestamp(w_start)
                end_str = format_ass_timestamp(w_end)

                # Assemble sentence with active word highlighted and bouncing
                formatted_words = []
                for idx, w in enumerate(words):
                    word_text = w.word.strip()
                    if idx == active_idx:
                        # Active word: Yellow color, 110% font bounce scale
                        formatted_words.append(
                            f"{{\\c{highlight_color}\\fscx110\\fscy110}}{word_text}{{\\r}}"
                        )
                    else:
                        # Inactive words: Bold white, standard 100% scale
                        formatted_words.append(
                            f"{{\\c{primary_color}\\fscx100\\fscy100}}{word_text}{{\\r}}"
                        )

                line_text = " ".join(formatted_words)
                lines.append(f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{line_text}")

        return "\n".join(lines) + "\n"

    def write_ass_file(
        self,
        output_path: Path,
        segments: List[TranscriptSegment],
        clip_start: float = 0.0,
        clip_end: Optional[float] = None,
        is_split_screen: bool = False,
    ) -> Path:
        """Write ASS subtitle script to disk."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        content = self.generate_ass_content(
            segments=segments,
            clip_start=clip_start,
            clip_end=clip_end,
            is_split_screen=is_split_screen,
        )
        output_path.write_text(content, encoding="utf-8")
        logger.debug("Wrote ASS subtitle script", path=str(output_path), size=len(content))
        return output_path
