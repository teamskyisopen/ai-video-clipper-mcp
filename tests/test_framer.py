"""Pure mathematical and invariant unit tests for SmartVideoFramer and FramingCoordinate."""

from pathlib import Path
import random
import pytest

from core.config import Settings
from domain.models import FramingCoordinate
from services.framer import (
    FaceObservation,
    SceneShot,
    SmartVideoFramer,
    VideoFramer,
)


class TestFramingCoordinateModel:
    """Validate FramingCoordinate schema and fields."""

    def test_model_instantiation(self) -> None:
        coord = FramingCoordinate(
            timestamp=2.5,
            crop_x=650,
            crop_y=0,
            width=608,
            height=1080,
            speaker_count=1,
        )
        assert coord.timestamp == 2.5
        assert coord.crop_x == 650
        assert coord.crop_y == 0
        assert coord.width == 608
        assert coord.height == 1080
        assert coord.speaker_count == 1
        assert coord.is_split is False

    def test_model_split_screen_flag(self) -> None:
        coord = FramingCoordinate(
            timestamp=4.0,
            crop_x=120,
            crop_y=50,
            width=972,
            height=864,
            speaker_count=2,
            is_split=True,
        )
        assert coord.is_split is True
        assert coord.speaker_count == 2


class TestSmartVideoFramerMath:
    """Validate pure mathematical invariants: 0 <= crop_x <= width - crop_w and 0 <= crop_y <= height - crop_h."""

    def test_alias_compatibility(self) -> None:
        assert VideoFramer is SmartVideoFramer

    def test_single_crop_dimension_aspect_ratio(self) -> None:
        settings = Settings(target_width=1080, target_height=1920)
        framer = SmartVideoFramer(settings)

        # Standard 16:9 full HD source
        source_w, source_h = 1920, 1080
        crop_w, crop_h = framer.calculate_single_crop_dimensions(source_w, source_h)

        assert crop_h == 1080
        # 1080 * (1080 / 1920) = 607.5 -> even integer 606 or 608
        assert crop_w in (606, 608)
        assert crop_w % 2 == 0
        assert crop_h % 2 == 0
        assert crop_w <= source_w
        assert crop_h <= source_h

    def test_extreme_boundary_clamping_invariants(self) -> None:
        framer = SmartVideoFramer()
        source_w, source_h = 1920, 1080
        crop_w, crop_h = 608, 1080

        # Highly negative inputs (e.g. face far off to the left/top)
        clamp_x, clamp_y = framer.clamp_crop_coordinates(
            raw_x=-1200, raw_y=-800,
            crop_w=crop_w, crop_h=crop_h,
            source_w=source_w, source_h=source_h,
        )
        assert clamp_x == 0
        assert clamp_y == 0
        assert 0 <= clamp_x <= (source_w - crop_w)
        assert 0 <= clamp_y <= (source_h - crop_h)

        # Excessively large inputs (e.g. face far off to the right/bottom)
        clamp_x, clamp_y = framer.clamp_crop_coordinates(
            raw_x=50000, raw_y=25000,
            crop_w=crop_w, crop_h=crop_h,
            source_w=source_w, source_h=source_h,
        )
        assert clamp_x == (source_w - crop_w) - ((source_w - crop_w) % 2)
        assert clamp_y == (source_h - crop_h) - ((source_h - crop_h) % 2)
        assert 0 <= clamp_x <= (source_w - crop_w)
        assert 0 <= clamp_y <= (source_h - crop_h)

    def test_randomized_boundary_invariance_fuzzing(self) -> None:
        """Fuzz test 500 arbitrary coordinates to verify invariant strictly holds."""
        framer = SmartVideoFramer()
        source_w, source_h = 1920, 1080
        crop_w, crop_h = 608, 1080

        random.seed(42)
        for _ in range(500):
            raw_x = random.randint(-5000, 10000)
            raw_y = random.randint(-5000, 10000)

            clamped_x, clamped_y = framer.clamp_crop_coordinates(
                raw_x=raw_x, raw_y=raw_y,
                crop_w=crop_w, crop_h=crop_h,
                source_w=source_w, source_h=source_h,
            )

            # Strict Invariants
            assert 0 <= clamped_x <= (source_w - crop_w)
            assert 0 <= clamped_y <= (source_h - crop_h)
            assert (clamped_x + crop_w) <= source_w
            assert (clamped_y + crop_h) <= source_h
            assert clamped_x % 2 == 0
            assert clamped_y % 2 == 0

    def test_ema_smoothing_resets_on_scene_cut(self, tmp_path: Path) -> None:
        """Verify EMA smoothing applies inside a scene, but resets across scene cuts."""
        settings = Settings(framer_smoothing_alpha=0.5)

        # Mock scenes: Scene 1 [0s - 3s], Scene 2 [3s - 6s]
        mock_scenes = [SceneShot(0.0, 3.0), SceneShot(3.0, 6.0)]

        # Speaker abruptly jumps from x=200 to x=1800 at scene boundary
        def mock_face_detector(video_path: Path, timestamp: float) -> list[FaceObservation]:
            if timestamp < 3.0:
                return [FaceObservation(cx=200.0, cy=500.0, width=120.0, height=120.0)]
            else:
                return [FaceObservation(cx=1800.0, cy=500.0, width=120.0, height=120.0)]

        framer = SmartVideoFramer(settings, face_detector=mock_face_detector)
        # Mock scene detection
        framer.detect_scenes = lambda video_path, s, e: mock_scenes  # type: ignore

        dummy_video = tmp_path / "test.mp4"
        dummy_video.write_bytes(b"dummy")

        trajectory = framer.calculate_framing_trajectory(
            dummy_video, start_time=0.0, end_time=6.0, source_width=1920, source_height=1080
        )

        scene1_coords = [c for c in trajectory if c.timestamp < 3.0]
        scene2_coords = [c for c in trajectory if c.timestamp >= 3.0]

        # In scene 1, speaker was at left: crop_x should be near 0
        assert scene1_coords[0].crop_x == 0

        # At the very start of scene 2, EMA MUST NOT lag or slide across cut from x=200:
        # It must instantly anchor to x=1800 (max right crop_x = 1920 - crop_w)
        assert scene2_coords[0].crop_x == (1920 - scene2_coords[0].width)

    def test_dual_speaker_split_screen_crops(self) -> None:
        """Validate dual-speaker split screen calculations and boundary conformance."""
        framer = SmartVideoFramer()
        source_w, source_h = 1920, 1080

        # Left speaker at cx=350, Right speaker at cx=1550 (separation = 1200 / 1920 = 0.625 > 0.20)
        face_left = FaceObservation(cx=350.0, cy=450.0, width=140.0, height=140.0)
        face_right = FaceObservation(cx=1550.0, cy=480.0, width=140.0, height=140.0)

        (top_x, top_y, w, h), (bot_x, bot_y, _, _) = framer.calculate_split_crops(
            face_left, face_right, source_w, source_h
        )

        # Invariant checks for both half-frame crop windows
        assert 0 <= top_x <= (source_w - w)
        assert 0 <= top_y <= (source_h - h)
        assert (top_x + w) <= source_w
        assert (top_y + h) <= source_h

        assert 0 <= bot_x <= (source_w - w)
        assert 0 <= bot_y <= (source_h - h)
        assert (bot_x + w) <= source_w
        assert (bot_y + h) <= source_h

        # Top speaker is on the left, so top_x must be to the left of bot_x
        assert top_x < bot_x
