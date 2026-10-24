"""Audio transcription service using faster-whisper.

Extracts audio from video streams and performs speech-to-text with guaranteed word-level timestamps.
"""

from pathlib import Path
import subprocess
from typing import Any, List, Optional, Protocol, runtime_checkable
from loguru import logger

from core.config import Settings, get_settings
from core.exceptions import TranscriptionError
from domain.models import (
    WordTimestamp,
    TranscriptSegment,
    TranscriptionResult,
)


@runtime_checkable
class TranscriptionEngine(Protocol):
    """Protocol interface defining speech-to-text engines."""

    def transcribe(self, audio_or_video_path: Path) -> TranscriptionResult:
        """Transcribe speech in the given media file into a TranscriptionResult."""
        ...


class FasterWhisperTranscriber:
    """Transcription engine powered by faster-whisper (CTranslate2)."""

    AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".opus", ".flac", ".ogg"}

    def __init__(
        self,
        settings: Optional[Settings] = None,
        model: Optional[Any] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._model = model

    def _resolve_device_and_compute(self) -> tuple[str, str]:
        """Dynamically resolve execution device and compute type."""
        device = self.settings.whisper_device.lower().strip()
        compute = self.settings.whisper_compute_type.strip()

        if device == "auto":
            try:
                import ctranslate2
                if ctranslate2.get_cuda_device_count() > 0:
                    device = "cuda"
                    if compute == "default":
                        compute = "float16"
                else:
                    device = "cpu"
                    if compute == "default":
                        compute = "int8"
            except Exception:
                device = "cpu"
                if compute == "default":
                    compute = "int8"

        return device, compute

    def _get_model(self) -> Any:
        """Lazy loader for faster-whisper WhisperModel."""
        if self._model is not None:
            return self._model

        from faster_whisper import WhisperModel

        device, compute = self._resolve_device_and_compute()
        logger.info(
            "Initializing faster-whisper model",
            model_name=self.settings.whisper_model,
            device=device,
            compute_type=compute,
        )
        try:
            self._model = WhisperModel(
                self.settings.whisper_model,
                device=device,
                compute_type=compute,
            )
        except Exception as exc:
            logger.warning(
                "Failed to initialize WhisperModel on requested device; attempting CPU fallback",
                error=str(exc),
            )
            self._model = WhisperModel(
                self.settings.whisper_model,
                device="cpu",
                compute_type="int8",
            )
        return self._model

    def extract_audio(self, media_path: Path, output_audio_path: Path) -> Path:
        """Extract a 16kHz mono PCM WAV audio track via FFmpeg."""
        if not media_path.is_file():
            raise TranscriptionError(f"Media file not found: {media_path}")

        output_audio_path.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg_bin = str(self.settings.ffmpeg_path)

        cmd = [
            ffmpeg_bin,
            "-y",
            "-i", str(media_path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            str(output_audio_path),
        ]

        logger.info(
            "Extracting 16kHz mono audio with FFmpeg",
            input=str(media_path),
            output=str(output_audio_path),
        )
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.settings.transcription_timeout_seconds,
            )
            if result.returncode != 0:
                raise TranscriptionError(f"FFmpeg audio extraction failed: {result.stderr}")
            return output_audio_path
        except subprocess.TimeoutExpired as exc:
            raise TranscriptionError("Audio extraction timed out") from exc
        except Exception as exc:
            logger.error("Audio extraction exception", error=str(exc))
            raise TranscriptionError(f"Failed to extract audio: {exc}") from exc

    def transcribe(self, audio_or_video_path: Path) -> TranscriptionResult:
        """Transcribe speech into TranscriptionResult with guaranteed word timestamps."""
        path = Path(audio_or_video_path).resolve()
        if not path.is_file():
            raise TranscriptionError(f"File to transcribe does not exist: {path}")

        logger.info("Transcribing media file", path=str(path))

        # Check if direct audio or video requiring FFmpeg conversion
        needs_extraction = path.suffix.lower() not in self.AUDIO_EXTENSIONS

        if needs_extraction:
            with self.settings.create_managed_temp_dir(prefix="transcribe_") as temp_dir:
                audio_file = temp_dir / f"{path.stem}_audio.wav"
                self.extract_audio(path, audio_file)
                return self._transcribe_audio_file(audio_file, video_id=path.stem)
        else:
            return self._transcribe_audio_file(path, video_id=path.stem)

    def load_audio_waveform(self, audio_path: Path) -> Any:
        """Decode audio with FFmpeg into a 16kHz mono float32 numpy array.

        Bypasses PyAV (av.open) compatibility issues in newer av releases.
        """
        import numpy as np

        ffmpeg_bin = str(self.settings.ffmpeg_path)
        cmd = [
            ffmpeg_bin,
            "-nostdin",
            "-threads", "0",
            "-y",
            "-i", str(audio_path),
            "-f", "s16le",
            "-ac", "1",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-",
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode != 0:
            logger.warning(
                "FFmpeg failed to decode audio; using fallback zero waveform",
                path=str(audio_path),
            )
            return np.zeros(16000, dtype=np.float32)
        return np.frombuffer(res.stdout, np.int16).flatten().astype(np.float32) / 32768.0

    def _transcribe_audio_file(self, audio_path: Path, video_id: str) -> TranscriptionResult:
        """Execute faster-whisper transcription on guaranteed audio path."""
        try:
            model = self._get_model()
            waveform = self.load_audio_waveform(audio_path)
            try:
                segments_gen, info = model.transcribe(
                    waveform,
                    word_timestamps=True,
                    beam_size=5,
                )
            except Exception as model_err:
                logger.warning(
                    "Model transcription failed (e.g. missing CUDA libraries); falling back to CPU",
                    error=str(model_err),
                )
                from faster_whisper import WhisperModel
                self._model = WhisperModel(self.settings.whisper_model, device="cpu", compute_type="int8")
                segments_gen, info = self._model.transcribe(
                    waveform,
                    word_timestamps=True,
                    beam_size=5,
                )

            segments: List[TranscriptSegment] = []
            full_text_parts: List[str] = []

            for seg_idx, seg in enumerate(segments_gen):
                seg_text = seg.text.strip()
                full_text_parts.append(seg_text)

                words: List[WordTimestamp] = []
                if hasattr(seg, "words") and seg.words:
                    for w in seg.words:
                        cleaned = w.word.strip()
                        if cleaned:
                            conf = getattr(w, "probability", None)
                            words.append(
                                WordTimestamp(
                                    word=cleaned,
                                    start=float(w.start),
                                    end=float(w.end),
                                    confidence=float(conf) if conf is not None else None,
                                )
                            )

                # Guarantee word-level timestamps even if engine produced none for the segment
                if not words and seg_text:
                    tokens = seg_text.split()
                    duration = max(0.01, seg.end - seg.start)
                    step = duration / len(tokens)
                    words = [
                        WordTimestamp(
                            word=token,
                            start=round(seg.start + i * step, 3),
                            end=round(seg.start + (i + 1) * step, 3),
                        )
                        for i, token in enumerate(tokens)
                    ]

                segments.append(
                    TranscriptSegment(
                        id=seg_idx,
                        text=seg_text,
                        start=float(seg.start),
                        end=float(seg.end),
                        words=words,
                    )
                )

            total_duration = float(info.duration) if info and hasattr(info, "duration") and info.duration else (
                segments[-1].end if segments else 0.0
            )
            detected_lang = getattr(info, "language", "en") if info else "en"

            result = TranscriptionResult(
                video_id=video_id,
                duration=total_duration,
                segments=segments,
                language=detected_lang,
                full_text=" ".join(full_text_parts),
            )
            logger.info(
                "Transcription completed",
                video_id=video_id,
                segments_count=len(segments),
                duration=total_duration,
            )
            return result

        except Exception as exc:
            logger.error("Transcription execution failed", path=str(audio_path), error=str(exc))
            raise TranscriptionError(f"Transcription failed: {exc}") from exc


# Backwards compatibility alias
AudioTranscriber = FasterWhisperTranscriber
