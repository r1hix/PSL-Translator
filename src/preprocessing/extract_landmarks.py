"""MediaPipe Holistic 55-keypoint landmark extraction pipeline.

Academic Explainability:
    Extracts skeletal coordinates (x, y, z) across 55 anatomical joints:
    - Left Hand: 21 keypoints (indices 0 to 20)
    - Right Hand: 21 keypoints (indices 21 to 41)
    - Upper Body Pose: 13 keypoints (indices 42 to 54)
    Total skeletal vertices V = 55.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HolisticLandmarker,
    HolisticLandmarkerOptions,
    RunningMode,
)
import numpy as np

from src.config import Config, load_config

logger = logging.getLogger("LandmarkExtractor")

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task"
DEFAULT_POSE_INDICES = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24]


def ensure_model_asset(model_path: Optional[Path] = None) -> Path:
    """Ensures the MediaPipe holistic_landmarker.task model asset exists locally.
    
    Args:
        model_path: Optional explicit Path to task asset.
        
    Returns:
        Path: Verified Path to holistic_landmarker.task.
    """
    if model_path is None:
        model_path = Path(__file__).resolve().parent.parent.parent / "models" / "checkpoints" / "holistic_landmarker.task"

    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading official MediaPipe Holistic model bundle...")
        urllib.request.urlretrieve(MODEL_URL, model_path)
        logger.info("Saved model asset to %s (%d bytes)", model_path, model_path.stat().st_size)

    return model_path


class LandmarkExtractor:
    """Extracts 55 skeletal landmarks from RGB image frames or video streams."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        pose_indices: Optional[List[int]] = None,
        min_detection_confidence: float = 0.5,
    ) -> None:
        """Initializes the HolisticLandmarker instance.
        
        Args:
            model_path: Optional path to holistic_landmarker.task.
            pose_indices: List of 13 pose indices to extract.
            min_detection_confidence: Detection threshold.
        """
        self.model_path = ensure_model_asset(model_path)
        self.pose_indices = pose_indices or DEFAULT_POSE_INDICES

        options = HolisticLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=RunningMode.IMAGE,
            output_face_blendshapes=False,
            output_segmentation_mask=False,
            min_hand_landmarks_confidence=min_detection_confidence,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_landmarks_confidence=min_detection_confidence,
        )
        self.landmarker = HolisticLandmarker.create_from_options(options)
        self.standard_dims = (1280, 720)

    def extract_frame_landmarks(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Extracts 55 skeletal keypoints from a single BGR/RGB frame.
        
        Keypoint Index Mapping:
            - [0:21]   : Left Hand (21 joints: wrist, thumb, index, middle, ring, pinky)
            - [21:42]  : Right Hand (21 joints)
            - [42:55]  : Upper Body Pose (13 joints: nose, eyes, ears, shoulders, elbows, wrists, hips)
            
        Args:
            frame_bgr: Video frame of shape (H, W, 3) in BGR or RGB format.
            
        Returns:
            np.ndarray: Landmark matrix of shape (55, 3) containing float32 (x, y, z).
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return np.zeros((55, 3), dtype=np.float32)

        # Standardize frame resolution to prevent internal MediaPipe calculator dimension conflicts
        h, w = frame_bgr.shape[:2]
        if (w, h) != self.standard_dims:
            frame_bgr = cv2.resize(frame_bgr, self.standard_dims)

        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = self.landmarker.detect(mp_image)

        # 1. Left Hand (21 joints)
        left_hand = np.zeros((21, 3), dtype=np.float32)
        if result.left_hand_landmarks:
            for i, lm in enumerate(result.left_hand_landmarks):
                left_hand[i] = [lm.x, lm.y, lm.z]

        # 2. Right Hand (21 joints)
        right_hand = np.zeros((21, 3), dtype=np.float32)
        if result.right_hand_landmarks:
            for i, lm in enumerate(result.right_hand_landmarks):
                right_hand[i] = [lm.x, lm.y, lm.z]

        # 3. Upper Body Pose (13 selected joints)
        upper_pose = np.zeros((13, 3), dtype=np.float32)
        if result.pose_landmarks:
            for i, p_idx in enumerate(self.pose_indices):
                if p_idx < len(result.pose_landmarks):
                    lm = result.pose_landmarks[p_idx]
                    upper_pose[i] = [lm.x, lm.y, lm.z]

        # Concatenate: (21 + 21 + 13, 3) = (55, 3)
        landmarks_55 = np.vstack([left_hand, right_hand, upper_pose])
        return landmarks_55.astype(np.float32)

    def extract_video_landmarks(
        self,
        video_path: Union[str, Path],
        max_extract_frames: int = 60,
        skip_empty_leads: bool = True,
    ) -> np.ndarray:
        """Extracts landmark coordinate sequences from an entire video file.
        
        Args:
            video_path: Path to input .mp4 video file.
            max_extract_frames: Maximum frames to extract across video (uniform temporal stride).
            skip_empty_leads: Whether to strip leading and trailing inactive frames.
            
        Returns:
            np.ndarray: Sequence of shape (F, 55, 3) where F is the number of valid frames.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Failed to open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return np.zeros((0, 55, 3), dtype=np.float32)

        # Compute frame indices to sample uniformly across video
        if total_frames <= max_extract_frames:
            sample_indices = set(range(total_frames))
        else:
            sample_indices = set(np.linspace(0, total_frames - 1, num=max_extract_frames, dtype=int))

        sequence: List[np.ndarray] = []
        f_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if f_idx in sample_indices:
                    landmarks = self.extract_frame_landmarks(frame)
                    sequence.append(landmarks)
                f_idx += 1
        finally:
            cap.release()


        if not sequence:
            return np.zeros((0, 55, 3), dtype=np.float32)

        arr = np.array(sequence, dtype=np.float32)  # Shape: (F, 55, 3)

        if skip_empty_leads and len(arr) > 0:
            # Detect frames where at least one pose/hand landmark is detected (non-zero norm)
            frame_activity = np.linalg.norm(arr, axis=(1, 2)) > 1e-4
            valid_indices = np.where(frame_activity)[0]
            if len(valid_indices) >= 5:
                start_f = valid_indices[0]
                end_f = valid_indices[-1] + 1
                arr = arr[start_f:end_f]

        return arr

    def close(self) -> None:
        """Closes the underlying MediaPipe landmarker and releases resources."""
        if hasattr(self, "landmarker") and self.landmarker:
            self.landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
