"""Unit tests for media ingestion, domain models, downloader, and transcription engine."""

from pathlib import Path
import struct
from unittest.mock import MagicMock, patch
import wave
import pytest
from pydantic import ValidationError

from core.config import Settings
from domain.models import (
    WordTimestamp,
    TranscriptSegment,
    TranscriptionResult,
)
from services.downloader import MediaDownloader, VideoDownloader
from services.transcriber import (
    FasterWhisperTranscriber,
    AudioTranscriber,
    TranscriptionEngine,
)


@pytest.fixture
def mock_audio_wav(tmp_path: Path) -> Path:
    """Generate a valid 1-second 16kHz mono PCM WAV fixture."""
    wav_path = tmp_path / "test_sample.wav"
    sample_rate = 16000
    duration_s = 1.0
    num_samples = int(sample_rate * duration_s)

    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        # Write silence / zero PCM samples
        raw_pcm = struct.pack(f"<{num_samples}h", *(0 for _ in range(num_samples)))
        wf.writeframes(raw_pcm)

    return wav_path


class TestDomainModels:
    """Validate strict Pydantic schemas for audio and transcript data."""

    def test_word_timestamp_success(self) -> None:
        word = WordTimestamp(word="Antigravity", start=0.12, end=0.85, confidence=0.98)
        assert word.word == "Antigravity"
        assert word.start == 0.12
        assert word.end == 0.85
        assert word.confidence == 0.98

    def test_word_timestamp_validation(self) -> None:
        with pytest.raises(ValidationError):
            WordTimestamp(word="Invalid", start=-1.0, end=1.0)

    def test_transcript_segment_success(self) -> None:
        words = [
            WordTimestamp(word="AI", start=0.0, end=0.3),
            WordTimestamp(word="Clipper", start=0.35, end=0.9),
        ]
        segment = TranscriptSegment(
            id=0,
            text="AI Clipper",
            start=0.0,
            end=0.9,
            words=words,
        )
        assert segment.id == 0
        assert segment.text == "AI Clipper"
        assert len(segment.words) == 2

    def test_transcription_result_success(self) -> None:
        words = [WordTimestamp(word="Test", start=0.0, end=0.5)]
        segments = [
            TranscriptSegment(id=0, text="Test", start=0.0, end=0.5, words=words)
        ]
        result = TranscriptionResult(
            video_id="vid_123",
            duration=0.5,
            segments=segments,
            language="en",
            full_text="Test",
        )
        assert result.video_id == "vid_123"
        assert result.duration == 0.5
        assert result.duration_seconds == 0.5
        assert len(result.segments) == 1


class TestMediaDownloader:
    """Validate MediaDownloader options building, audio extraction, and slicing."""

    def test_alias_compatibility(self) -> None:
        assert VideoDownloader is MediaDownloader

    def test_base_ydl_opts_injection(self) -> None:
        settings = Settings(
            proxy="http://127.0.0.1:8080",
            download_rate_limit="2M",
        )
        downloader = MediaDownloader(settings)
        opts = downloader._build_base_ydl_opts()

        assert opts["proxy"] == "http://127.0.0.1:8080"
        assert opts["ratelimit"] == 2097152  # 2MB in bytes
        assert opts["quiet"] is True

    @patch("yt_dlp.YoutubeDL")
    def test_download_audio_options(self, mock_ydl_cls: MagicMock, tmp_path: Path) -> None:
        settings = Settings(
            temp_storage_dir=tmp_path,
            audio_codec="mp3",
            audio_bitrate="128k",
            proxy="socks5://localhost:1080",
        )
        downloader = MediaDownloader(settings)

        # Mock YoutubeDL instance
        mock_ydl = MagicMock()
        mock_ydl_cls.return_value.__enter__.return_value = mock_ydl

        fake_raw = tmp_path / "audio123.webm"
        fake_audio = tmp_path / "audio123.mp3"
        fake_audio.write_bytes(b"mock mp3 data")

        mock_ydl.extract_info.return_value = {"id": "audio123", "title": "Test Audio"}
        mock_ydl.prepare_filename.return_value = str(fake_raw)

        result_path = downloader.download_audio("https://youtube.com/watch?v=audio123")

        # Verify YoutubeDL was instantiated with audio extraction postprocessor
        call_opts = mock_ydl_cls.call_args[0][0]
        assert call_opts["proxy"] == "socks5://localhost:1080"
        assert call_opts["format"] == "ba/b"
        assert any(
            p.get("key") == "FFmpegExtractAudio" and p.get("preferredcodec") == "mp3"
            for p in call_opts["postprocessors"]
        )
        assert result_path == fake_audio.resolve()

    @patch("yt_dlp.YoutubeDL")
    def test_download_video_slice_options(self, mock_ydl_cls: MagicMock, tmp_path: Path) -> None:
        settings = Settings(temp_storage_dir=tmp_path)
        downloader = MediaDownloader(settings)

        mock_ydl = MagicMock()
        mock_ydl_cls.return_value.__enter__.return_value = mock_ydl

        fake_slice = tmp_path / "slice123.mp4"
        fake_slice.write_bytes(b"mock mp4 slice")

        mock_ydl.extract_info.return_value = {"id": "vid123"}
        mock_ydl.prepare_filename.return_value = str(fake_slice)

        result_path = downloader.download_video_slice(
            url="https://youtube.com/watch?v=vid123",
            start=10.0,
            end=30.0,
        )

        call_opts = mock_ydl_cls.call_args[0][0]
        assert call_opts["force_keyframes_at_cuts"] is True
        assert "download_ranges" in call_opts
        assert result_path == fake_slice.resolve()


class TestFasterWhisperTranscriber:
    """Validate FasterWhisperTranscriber engine, word-level guarantee, and FFmpeg audio extraction."""

    def test_engine_protocol_conformance(self) -> None:
        transcriber = FasterWhisperTranscriber()
        assert isinstance(transcriber, TranscriptionEngine)
        assert AudioTranscriber is FasterWhisperTranscriber

    def test_device_resolution(self) -> None:
        settings_cpu = Settings(whisper_device="cpu", whisper_compute_type="int8")
        transcriber = FasterWhisperTranscriber(settings_cpu)
        dev, comp = transcriber._resolve_device_and_compute()
        assert dev == "cpu"
        assert comp == "int8"

    def test_transcription_guarantees_word_timestamps(self, tmp_path: Path) -> None:
        """Verify that segments without word timestamps get synthesized word-level timestamps."""
        fake_audio = tmp_path / "speech.wav"
        fake_audio.write_bytes(b"RIFF dummy wav")

        # Mock segment without words
        class FakeSegment:
            id = 0
            start = 1.0
            end = 3.0
            text = "Automated video clipping"
            words = None

        class FakeInfo:
            duration = 3.0
            language = "en"

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([FakeSegment()], FakeInfo())

        transcriber = FasterWhisperTranscriber(model=mock_model)
        result = transcriber.transcribe(fake_audio)

        assert isinstance(result, TranscriptionResult)
        assert len(result.segments) == 1
        seg = result.segments[0]
        assert seg.text == "Automated video clipping"

        # Check synthesized word timestamps are guaranteed
        assert len(seg.words) == 3
        words = [w.word for w in seg.words]
        assert words == ["Automated", "video", "clipping"]
        assert seg.words[0].start == 1.0
        assert seg.words[-1].end == 3.0

    def test_transcription_with_explicit_words(self, tmp_path: Path) -> None:
        """Verify that faster-whisper native word timestamps are mapped properly."""
        fake_audio = tmp_path / "speech.wav"
        fake_audio.write_bytes(b"RIFF dummy wav")

        class FakeWord:
            def __init__(self, word: str, start: float, end: float, probability: float):
                self.word = word
                self.start = start
                self.end = end
                self.probability = probability

        class FakeSegment:
            id = 0
            start = 0.5
            end = 2.0
            text = "Hello world"
            words = [
                FakeWord(" Hello", 0.5, 1.0, 0.95),
                FakeWord(" world", 1.1, 2.0, 0.99),
            ]

        class FakeInfo:
            duration = 2.0
            language = "en"

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([FakeSegment()], FakeInfo())

        transcriber = FasterWhisperTranscriber(model=mock_model)
        result = transcriber.transcribe(fake_audio)

        assert len(result.segments[0].words) == 2
        w1, w2 = result.segments[0].words
        assert w1.word == "Hello"
        assert w1.start == 0.5
        assert w1.confidence == 0.95
        assert w2.word == "world"
        assert w2.end == 2.0

    def test_real_audio_extraction_with_ffmpeg(self, mock_audio_wav: Path, tmp_path: Path) -> None:
        """Test real audio extraction using the local FFmpeg binary on the mock audio fixture."""
        output_wav = tmp_path / "extracted_16k.wav"
        transcriber = FasterWhisperTranscriber()

        extracted = transcriber.extract_audio(mock_audio_wav, output_wav)
        assert extracted.is_file()
        assert extracted.stat().st_size > 0

        # Validate that the extracted audio is a valid 16kHz mono WAV file
        with wave.open(str(extracted), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getframerate() == 16000
            assert wf.getsampwidth() == 2
