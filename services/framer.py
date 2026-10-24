"""Smart video framing, scene cut detection, and speaker tracking service.

Handles dynamic 16:9 to 9:16 vertical crop calculation, EMA smoothing across scene cuts,
and dual-speaker stacked split-screen layout math.
"""

from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
from typing import Callable, List, Optional, Tuple
from loguru import logger

from core.config import Settings, get_settings
from core.exceptions import FramingError
from domain.models import (
    CropWindow,
    FramingCoordinate,
    FramingKeyframe,
    VideoMetadata,
)


@dataclass(frozen=True)
class SceneShot:
    """Temporal boundaries of a single continuous scene shot."""
    start_s: float
    end_s: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end_s - self.start_s)


@dataclass(frozen=True)
class FaceObservation:
    """Bounding box and center coordinates of a detected face."""
    cx: float
    cy: float
    width: float
    height: float
    confidence: float = 1.0


class SmartVideoFramer:
    """Calculates intelligent vertical framing, scene cut tracking, and split layouts."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        face_detector: Optional[Callable[[Path, float], List[FaceObservation]]] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._custom_face_detector = face_detector

    # =========================================================================
    # Step 1: Scene-Cut Detection
    # =========================================================================

    def detect_scenes(
        self,
        video_path: Path,
        start_time: float = 0.0,
        end_time: Optional[float] = None,
    ) -> List[SceneShot]:
        """Detect scene cut boundaries across the video slice using PySceneDetect or FFmpeg."""
        if not video_path.is_file():
            raise FramingError(f"Video file does not exist: {video_path}")

        # Try PySceneDetect if installed
        try:
            return self._detect_scenes_scenedetect(video_path, start_time, end_time)
        except Exception as exc:
            logger.debug("PySceneDetect not available or failed; using FFmpeg scene filter", error=str(exc))

        # Fallback to FFmpeg scene filter
        return self._detect_scenes_ffmpeg(video_path, start_time, end_time)

    def _detect_scenes_scenedetect(
        self,
        video_path: Path,
        start_time: float,
        end_time: Optional[float],
    ) -> List[SceneShot]:
        """Detect scenes using scenedetect package."""
        import scenedetect
        from scenedetect import ContentDetector, SceneManager, open_video

        video = open_video(str(video_path))
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=self.settings.scene_detection_threshold))

        if start_time > 0:
            video.seek(start_time)

        scene_manager.detect_scenes(video=video, end_time=end_time)
        scene_list = scene_manager.get_scene_list()

        shots = []
        for start_tc, end_tc in scene_list:
            shots.append(SceneShot(start_s=start_tc.get_seconds(), end_s=end_tc.get_seconds()))

        return self._merge_short_shots(shots, start_time, end_time)

    def _detect_scenes_ffmpeg(
        self,
        video_path: Path,
        start_time: float,
        end_time: Optional[float],
    ) -> List[SceneShot]:
        """Detect scene cuts using FFmpeg scene filter via stderr timestamps."""
        ffmpeg_bin = str(self.settings.ffmpeg_path)
        threshold = self.settings.scene_detection_threshold / 100.0  # Normalized to 0.0-1.0

        duration = (end_time - start_time) if end_time else None
        cmd = [ffmpeg_bin, "-hide_banner", "-nostats", "-y"]
        if start_time > 0:
            cmd.extend(["-ss", str(start_time)])
        cmd.extend(["-i", str(video_path)])
        if duration:
            cmd.extend(["-t", str(duration)])
        cmd.extend([
            "-vf", f"select='gt(scene,{threshold:.2f})',showinfo",
            "-f", "null", "-",
        ])

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.settings.render_timeout_seconds,
            )
            raw_pts = re.findall(r"pts_time:(\d+\.?\d*)", result.stderr)
            cut_times = [start_time + float(t) for t in raw_pts]

            shots = []
            current_start = start_time
            for cut in cut_times:
                if cut > current_start:
                    shots.append(SceneShot(start_s=current_start, end_s=cut))
                    current_start = cut

            final_end = end_time if end_time else (current_start + 1.0)
            if final_end > current_start:
                shots.append(SceneShot(start_s=current_start, end_s=final_end))

            if not shots:
                shots = [SceneShot(start_s=start_time, end_s=final_end)]

            return self._merge_short_shots(shots, start_time, end_time)

        except Exception as exc:
            logger.warning("FFmpeg scene cut detection failed; using single shot", error=str(exc))
            final_end = end_time if end_time else (start_time + 10.0)
            return [SceneShot(start_s=start_time, end_s=final_end)]

    def _merge_short_shots(
        self,
        shots: List[SceneShot],
        start_time: float,
        end_time: Optional[float],
    ) -> List[SceneShot]:
        """Merge shots shorter than min_scene_duration into predecessor."""
        min_dur = self.settings.min_scene_duration_seconds
        if not shots:
            final_end = end_time if end_time else (start_time + 1.0)
            return [SceneShot(start_s=start_time, end_s=final_end)]

        merged: List[SceneShot] = []
        for shot in shots:
            if not merged:
                merged.append(shot)
            elif shot.duration < min_dur:
                # Merge into previous shot
                prev = merged.pop()
                merged.append(SceneShot(start_s=prev.start_s, end_s=shot.end_s))
            else:
                merged.append(shot)
        return merged

    # =========================================================================
    # Step 2: Frame Face Detection
    # =========================================================================

    def detect_faces(self, video_path: Path, timestamp: float) -> List[FaceObservation]:
        """Sample frame at timestamp and detect face bounding boxes."""
        if self._custom_face_detector is not None:
            return self._custom_face_detector(video_path, timestamp)

        # Try OpenCV Haar Cascade if installed
        try:
            import cv2
            cap = cv2.VideoCapture(str(video_path))
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
            ret, frame = cap.read()
            cap.release()

            if not ret or frame is None:
                return []

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            face_cascade = cv2.CascadeClassifier(cascade_path)
            detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            observations = []
            for (x, y, w, h) in detected:
                observations.append(
                    FaceObservation(
                        cx=float(x + w / 2.0),
                        cy=float(y + h / 2.0),
                        width=float(w),
                        height=float(h),
                    )
                )
            return observations
        except Exception:
            # Fallback when OpenCV is unconfigured
            return []

    # =========================================================================
    # Step 3 & 4: Crop Math, EMA Smoothing & Dual-Speaker Split Layout
    # =========================================================================

    def calculate_single_crop_dimensions(
        self, source_width: int, source_height: int
    ) -> Tuple[int, int]:
        """Calculate 9:16 portrait crop dimensions that fit strictly inside source dimensions."""
        target_aspect = self.settings.target_width / self.settings.target_height

        crop_height = source_height
        crop_width = int(crop_height * target_aspect)

        if crop_width > source_width:
            crop_width = source_width
            crop_height = int(crop_width / target_aspect)

        # Force even dimensions for FFmpeg encoder compliance
        crop_width -= crop_width % 2
        crop_height -= crop_height % 2

        # Clamp strictly
        crop_width = max(2, min(crop_width, source_width))
        crop_height = max(2, min(crop_height, source_height))

        return crop_width, crop_height

    def clamp_crop_coordinates(
        self,
        raw_x: int,
        raw_y: int,
        crop_w: int,
        crop_h: int,
        source_w: int,
        source_h: int,
    ) -> Tuple[int, int]:
        """Enforce strict invariant: 0 <= crop_x <= source_w - crop_w and 0 <= crop_y <= source_h - crop_h."""
        max_x = max(0, source_w - crop_w)
        max_y = max(0, source_h - crop_h)

        clamped_x = max(0, min(raw_x, max_x))
        clamped_y = max(0, min(raw_y, max_y))

        # Align to even integers
        clamped_x -= clamped_x % 2
        clamped_y -= clamped_y % 2

        return max(0, min(clamped_x, max_x)), max(0, min(clamped_y, max_y))

    def calculate_split_crops(
        self,
        face1: FaceObservation,
        face2: FaceObservation,
        source_w: int,
        source_h: int,
    ) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
        """Calculate dual stacked half-frame crop coordinates (top speaker, bottom speaker).

        Returns: ((top_x, top_y, w, h), (bot_x, bot_y, w, h))
        """
        # Sort left to right
        left_face, right_face = (face1, face2) if face1.cx <= face2.cx else (face2, face1)

        out_w = self.settings.target_width
        out_h = self.settings.target_height
        half_h = (out_h // 2) - ((out_h // 2) % 2)
        aspect = out_w / float(half_h)

        # Scale crop box relative to source height (0.8 tightness factor from openshorts)
        crop_h = int(round(source_h * 0.8))
        crop_h -= crop_h % 2
        crop_w = int(round(crop_h * aspect))
        crop_w -= crop_w % 2

        if crop_w > source_w:
            crop_w = source_w - (source_w % 2)
            crop_h = int(round(crop_w / aspect))
            crop_h -= crop_h % 2

        # Left speaker positioned on top
        raw_top_x = int(round(left_face.cx - crop_w / 2.0))
        raw_top_y = int(round(left_face.cy - crop_h * 0.42))
        top_x, top_y = self.clamp_crop_coordinates(raw_top_x, raw_top_y, crop_w, crop_h, source_w, source_h)

        # Right speaker positioned on bottom
        raw_bot_x = int(round(right_face.cx - crop_w / 2.0))
        raw_bot_y = int(round(right_face.cy - crop_h * 0.42))
        bot_x, bot_y = self.clamp_crop_coordinates(raw_bot_x, raw_bot_y, crop_w, crop_h, source_w, source_h)

        return (top_x, top_y, crop_w, crop_h), (bot_x, bot_y, crop_w, crop_h)

    def calculate_framing_trajectory(
        self,
        video_path: Path,
        start_time: float,
        end_time: float,
        source_width: int = 1920,
        source_height: int = 1080,
        sample_fps: float = 2.0,
    ) -> List[FramingCoordinate]:
        """Compute smooth framing coordinates across detected scenes with boundary invariance."""
        crop_w, crop_h = self.calculate_single_crop_dimensions(source_width, source_height)
        scenes = self.detect_scenes(video_path, start_time, end_time)

        trajectory: List[FramingCoordinate] = []
        alpha = self.settings.framer_smoothing_alpha

        for scene in scenes:
            scene_duration = scene.end_s - scene.start_s
            num_samples = max(2, int(scene_duration * sample_fps))
            step = scene_duration / float(num_samples)

            # Scene-cut boundary rule: Reset EMA filter state at start of each new scene
            smoothed_cx: Optional[float] = None

            for i in range(num_samples):
                t = min(scene.start_s + i * step, scene.end_s)
                faces = self.detect_faces(video_path, t)

                # Check if dual speaker split-screen criteria are met
                is_split = False
                if self.settings.split_screen_enabled and len(faces) >= 2:
                    face1, face2 = faces[0], faces[1]
                    separation_ratio = abs(face1.cx - face2.cx) / float(source_width)
                    if separation_ratio >= self.settings.split_min_separation:
                        is_split = True
                        (top_x, top_y, sw, sh), (bot_x, bot_y, _, _) = self.calculate_split_crops(
                            face1, face2, source_width, source_height
                        )
                        coord = FramingCoordinate(
                            timestamp=round(t, 2),
                            crop_x=top_x,
                            crop_y=top_y,
                            width=sw,
                            height=sh,
                            speaker_count=2,
                            is_split=True,
                        )
                        trajectory.append(coord)
                        continue

                # Single speaker or default centered framing
                if faces:
                    # Choose primary face (largest area)
                    primary_face = max(faces, key=lambda f: f.width * f.height)
                    raw_cx = primary_face.cx
                    raw_cy = primary_face.cy
                    speaker_count = len(faces)
                else:
                    raw_cx = float(source_width / 2.0)
                    raw_cy = float(source_height / 2.0)
                    speaker_count = 0

                # Apply Exponential Moving Average (EMA) inside this scene
                if smoothed_cx is None:
                    smoothed_cx = raw_cx
                else:
                    smoothed_cx = alpha * raw_cx + (1.0 - alpha) * smoothed_cx

                # Headroom rule (38% down from crop top)
                target_top_y = raw_cy - crop_h * 0.38 if faces else (source_height - crop_h) / 2.0
                target_left_x = smoothed_cx - crop_w / 2.0

                clamped_x, clamped_y = self.clamp_crop_coordinates(
                    int(round(target_left_x)),
                    int(round(target_top_y)),
                    crop_w,
                    crop_h,
                    source_width,
                    source_height,
                )

                coord = FramingCoordinate(
                    timestamp=round(t, 2),
                    crop_x=clamped_x,
                    crop_y=clamped_y,
                    width=crop_w,
                    height=crop_h,
                    speaker_count=speaker_count,
                    is_split=False,
                )
                trajectory.append(coord)

        return trajectory

    # =========================================================================
    # Backwards Compatibility
    # =========================================================================

    def calculate_center_crop(self, source_width: int, source_height: int) -> CropWindow:
        """Static centered crop window for compatibility."""
        crop_w, crop_h = self.calculate_single_crop_dimensions(source_width, source_height)
        crop_x = max(0, (source_width - crop_w) // 2)
        crop_y = max(0, (source_height - crop_h) // 2)
        return CropWindow(x=crop_x, y=crop_y, width=crop_w, height=crop_h)

    def generate_framing_track(
        self,
        metadata: VideoMetadata,
        start_time: float,
        end_time: float,
    ) -> List[FramingKeyframe]:
        """Compatibility track adapter."""
        coords = self.calculate_framing_trajectory(
            video_path=metadata.file_path,
            start_time=start_time,
            end_time=end_time,
            source_width=metadata.width,
            source_height=metadata.height,
        )
        keyframes = []
        for c in coords:
            keyframes.append(
                FramingKeyframe(
                    timestamp=c.timestamp,
                    active_speaker=c.speaker_count > 0,
                    crop_window=CropWindow(
                        x=c.crop_x,
                        y=c.crop_y,
                        width=c.width,
                        height=c.height,
                    ),
                )
            )
        return keyframes


# Backwards compatibility alias
VideoFramer = SmartVideoFramer
