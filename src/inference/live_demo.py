"""Real-Time Pakistani Sign Language (PSL) Live Demonstration & Bilingual HUD.

Academic Explainability:
    Maintains a 30-frame rolling FIFO buffer (1.0 sec window at 30 FPS) of 55 skeletal
    joints extracted via MediaPipe Holistic. Keypoints are centered relative to the chest root
    and scale-normalized by inter-shoulder Euclidean distance before being dispatched to
    the ST-GCN neural model.

    To eliminate auditory chatter and false positives, temporal confidence smoothing requires
    K consecutive consistent frames exceeding confidence threshold theta before triggering
    the asynchronous spoken Urdu audio engine and bilingual HUD banner.
"""

import argparse
from collections import deque
import logging
from pathlib import Path
import platform
import sys
import time
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from src.config import Config, load_config
from src.inference.audio_engine import AudioEngine
from src.inference.translation_layer import BilingualTranslator, LanguageMode, SentenceStyle, TranslatedOutput
from src.models.graph import SKELETAL_EDGES
from src.models.st_gcn import STGCN
from src.preprocessing.extract_landmarks import LandmarkExtractor
from src.preprocessing.normalize import normalize_sequence
from src.utils import get_device

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("LiveHUD")


def get_urdu_font(size: int = 34) -> Optional[ImageFont.FreeTypeFont]:
    """Finds and loads a system font capable of rendering Arabic/Urdu scripts.
    
    Args:
        size: Desired font point size.
        
    Returns:
        Optional[ImageFont.FreeTypeFont]: Initialized TrueType font or None.
    """
    candidates = [
        "/System/Library/Fonts/SFArabic.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/GeezaPro.ttc",
        "C:\\Windows\\Fonts\\arial.ttf",
        "C:\\Windows\\Fonts\\segoeui.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    try:
        return ImageFont.load_default()
    except Exception:
        return None


class RealTimePSLTranslator:
    """End-to-End Real-Time Sign Language Inference and HUD Interface."""

    def __init__(
        self,
        config: Optional[Config] = None,
        checkpoint_path: Optional[Path] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initializes the real-time inference loop, models, and audio engines.
        
        Args:
            config: Optional pre-loaded Config object.
            checkpoint_path: Path to trained PyTorch model weights (.pth).
            device: Compute device (MPS on Mac, CUDA on PC, or CPU).
        """
        self.config = config or load_config()
        self.device = device or get_device()
        logger.info("Initializing Real-Time PSL Engine on Compute Device: %s", self.device)

        # 1. Feature Extractor & Pipeline Parameters
        self.extractor = LandmarkExtractor()
        self.window_size: int = self.config.data.sequence_length  # 30 frames
        self.buffer: Deque[np.ndarray] = deque(maxlen=self.window_size)

        # 2. Model Initialization
        self.model = STGCN(
            in_channels=self.config.data.num_channels,
            num_classes=self.config.num_classes,
        ).to(self.device)
        self.model.eval()

        self.model_loaded = False
        ckpt = checkpoint_path or (self.config.paths.checkpoints / "st_gcn_best.pth")
        if ckpt.exists():
            try:
                state_dict = torch.load(ckpt, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model_loaded = True
                logger.info("Loaded trained model weights from: %s", ckpt)
            except Exception as e:
                logger.warning("Could not load weights from %s: %s", ckpt, e)
        else:
            logger.info("No checkpoint found at %s. Running in architecture demo mode.", ckpt)

        # 3. Audio & Translation Engines
        self.translator = BilingualTranslator(config=self.config)
        self.audio_engine = AudioEngine(config=self.config, cooldown_sec=2.5)

        # 4. HUD State Tracking
        self.confidence_threshold = self.config.inference.confidence_threshold
        self.consecutive_trigger_frames = self.config.inference.consecutive_frames_trigger
        self.recent_predictions: Deque[int] = deque(maxlen=self.consecutive_trigger_frames)

        self.active_output: Optional[TranslatedOutput] = None
        self.active_confidence: float = 0.0
        self.draw_debug_skeleton: bool = self.config.inference.draw_landmarks
        self.urdu_font = get_urdu_font(size=36)
        self.urdu_font_small = get_urdu_font(size=24)

    def process_frame(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, Optional[TranslatedOutput]]:
        """Extracts landmarks, evaluates rolling sequence buffer, and updates HUD overlay.
        
        Args:
            frame_bgr: Raw input camera frame of shape (H, W, 3).
            
        Returns:
            Tuple[np.ndarray, Optional[TranslatedOutput]]: Rendered frame and active prediction.
        """
        h, w = frame_bgr.shape[:2]
        display_frame = frame_bgr.copy()

        # 1. Extract 55 Keypoints (Left Hand: 21, Right Hand: 21, Pose: 13)
        landmarks = self.extractor.extract_frame_landmarks(frame_bgr)
        self.buffer.append(landmarks)

        # 2. Render Skeletal Topology Overlay if enabled
        if self.draw_debug_skeleton:
            self._draw_skeleton_overlay(display_frame, landmarks, w, h)

        # 3. Evaluate Neural Model when 30-frame FIFO buffer is saturated
        if len(self.buffer) == self.window_size:
            raw_seq = np.array(self.buffer, dtype=np.float32)  # (30, 55, 3)
            # Normalize sequence: chest root centering & shoulder span scaling
            norm_seq = normalize_sequence(raw_seq, target_length=self.window_size)
            # Reshape into PyTorch ST-GCN contract: (1, C=3, T=30, V=55)
            tensor_batch = (
                torch.from_numpy(norm_seq)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(self.device)
            )

            with torch.no_grad():
                logits = self.model(tensor_batch)
                probabilities = torch.softmax(logits, dim=-1)[0]
                conf, pred_idx = torch.max(probabilities, dim=-1)
                pred_class = int(pred_idx.item())
                confidence_score = float(conf.item())

            self.recent_predictions.append(pred_class)

            # Check consecutive trigger threshold for stability
            is_consistent = (
                len(self.recent_predictions) == self.consecutive_trigger_frames
                and all(p == pred_class for p in self.recent_predictions)
            )

            if is_consistent and confidence_score >= self.confidence_threshold:
                self.active_output = self.translator.translate(pred_class, confidence=confidence_score)
                self.active_confidence = confidence_score
                # Trigger asynchronous non-blocking audio pronunciation
                self.audio_engine.play_word(
                    self.active_output.word_name,
                    communicative=(self.translator.style == SentenceStyle.COMMUNICATIVE),
                    async_mode=True,
                )

        # 4. Render Bilingual Glassmorphic HUD Card
        rendered_frame = self._render_hud(display_frame, w, h)
        return rendered_frame, self.active_output

    def _draw_skeleton_overlay(self, frame: np.ndarray, landmarks: np.ndarray, width: int, height: int) -> None:
        """Draws skeletal nodes and anatomical bones onto the OpenCV frame."""
        # Draw bones
        for u, v in SKELETAL_EDGES:
            pt1 = landmarks[u]
            pt2 = landmarks[v]
            if np.linalg.norm(pt1) > 1e-4 and np.linalg.norm(pt2) > 1e-4:
                x1, y1 = int(pt1[0] * width), int(pt1[1] * height)
                x2, y2 = int(pt2[0] * width), int(pt2[1] * height)
                color = (0, 255, 200) if u < 42 and v < 42 else (255, 160, 50)
                cv2.line(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        # Draw joints
        for i in range(55):
            pt = landmarks[i]
            if np.linalg.norm(pt) > 1e-4:
                x, y = int(pt[0] * width), int(pt[1] * height)
                if i < 21:
                    cv2.circle(frame, (x, y), 3, (0, 255, 100), -1)  # Left hand
                elif i < 42:
                    cv2.circle(frame, (x, y), 3, (0, 180, 255), -1)  # Right hand
                else:
                    cv2.circle(frame, (x, y), 4, (255, 200, 0), -1)   # Pose

    def _render_hud(self, frame: np.ndarray, width: int, height: int) -> np.ndarray:
        """Composes a modern semi-transparent bilingual HUD banner using OpenCV and PIL."""
        overlay = frame.copy()

        # 1. Top Status Banner
        cv2.rectangle(overlay, (0, 0), (width, 50), (20, 20, 20), -1)

        # 2. Bottom Bilingual Translation Card (Height: 140px)
        card_h = 140
        card_y = height - card_h
        cv2.rectangle(overlay, (0, card_y), (width, height), (15, 17, 26), -1)
        cv2.line(overlay, (0, card_y), (width, card_y), (0, 230, 180), 2)

        # Blend semi-transparent glassmorphism
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Top Bar Text
        dev_str = f"RIG: {self.device.type.upper()}"
        mode_str = f"MODE: {self.translator.mode.value.upper()} | STYLE: {self.translator.style.value.upper()}"
        audio_str = f"AUDIO: {'ON' if self.audio_engine.enabled else 'MUTED'}"
        cv2.putText(frame, f"PSL-TRANSLATOR v{self.config.version} | {dev_str} | {mode_str} | {audio_str}", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

        buf_pct = min(100, int(len(self.buffer) / self.window_size * 100))
        cv2.putText(frame, f"Buffer: {buf_pct}%", (width - 140, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200) if buf_pct == 100 else (120, 120, 120), 1, cv2.LINE_AA)

        # Bottom Card Details
        if self.active_output is not None:
            # English Title / Roman Urdu
            title_text = f"SIGN: {self.active_output.word_name.upper()} ({self.active_output.roman_text}) -- Conf: {int(self.active_confidence * 100)}%"
            cv2.putText(frame, title_text, (30, card_y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 230, 180), 2, cv2.LINE_AA)

            # English Spoken Sentence
            eng_sub = f"English: \"{self.active_output.english_text}\""
            cv2.putText(frame, eng_sub, (30, card_y + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

            # Urdu Text Rendering with PIL FreeType Font
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            draw = ImageDraw.Draw(pil_img)

            urdu_display = f"اردو: {self.active_output.urdu_text}"
            if self.urdu_font:
                draw.text((30, card_y + 86), urdu_display, font=self.urdu_font, fill=(0, 255, 180))
            else:
                draw.text((30, card_y + 86), self.active_output.roman_text, fill=(0, 255, 180))

            frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        else:
            status_text = "READY: Perform a sign within camera view..." if self.model_loaded else "DEMO MODE: Model architecture active (Perform signing gestures)"
            cv2.putText(frame, status_text, (30, card_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (160, 160, 160), 1, cv2.LINE_AA)

        return frame


def run_live_webcam(
    webcam_index: int = 0,
    video_source: Optional[str] = None,
    checkpoint: Optional[str] = None,
    flip_camera: Optional[bool] = None,
) -> None:
    """Runs the live interactive OpenCV webcam inference application."""
    config = load_config()
    translator = RealTimePSLTranslator(config=config, checkpoint_path=Path(checkpoint) if checkpoint else None)

    source = video_source if video_source is not None else webcam_index
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        logger.error("Could not open video capture stream: %s", source)
        return

    # Request high resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.inference.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.inference.frame_height)

    window_name = f"Pakistani Sign Language (PSL) Real-Time Translator - [{platform.system()}]"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # By default, flip live webcam stream for natural mirror view; do not flip video files
    should_flip = (video_source is None) if flip_camera is None else flip_camera

    prev_time = time.time()
    fps = 0.0

    print("\n" + "=" * 80)
    print("  PSL REAL-TIME TRANSLATOR -- INTERACTIVE CONTROLS")
    print("=" * 80)
    print("  [Q] : Quit live demonstration")
    print("  [F] : Toggle Camera Horizontal Flip (Mirror View)")
    print("  [M] : Toggle Audio Mute / Unmute")
    print("  [L] : Toggle Language Mode (Bilingual -> Urdu -> English)")
    print("  [S] : Toggle Sentence Style (Communicative Sentence <-> Literal Word)")
    print("  [D] : Toggle Landmark Skeletal Wireframe Overlay")
    print("=" * 80 + "\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("Video stream ended or frame unavailable.")
                break

            # Apply horizontal mirror flip if enabled
            if should_flip:
                frame = cv2.flip(frame, 1)

            # Process frame through ST-GCN & HUD pipeline
            rendered, _ = translator.process_frame(frame)

            # Compute and overlay FPS
            curr_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-4, curr_time - prev_time))
            prev_time = curr_time
            flip_tag = " [FLIPPED]" if should_flip else ""
            cv2.putText(rendered, f"FPS: {fps:.1f}{flip_tag}", (rendered.shape[1] - 270, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 1, cv2.LINE_AA)

            cv2.imshow(window_name, rendered)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("f"):
                should_flip = not should_flip
                logger.info("Camera flip toggled: %s", "MIRROR (FLIPPED)" if should_flip else "NORMAL (UNFLIPPED)")
            elif key == ord("m"):
                translator.audio_engine.enabled = not translator.audio_engine.enabled
                logger.info("Audio toggled: %s", "ENABLED" if translator.audio_engine.enabled else "MUTED")
            elif key == ord("l"):
                modes = [LanguageMode.BILINGUAL, LanguageMode.URDU, LanguageMode.ENGLISH]
                cur_idx = modes.index(translator.translator.mode)
                translator.translator.mode = modes[(cur_idx + 1) % len(modes)]
                logger.info("Language mode: %s", translator.translator.mode.value)
            elif key == ord("s"):
                styles = [SentenceStyle.COMMUNICATIVE, SentenceStyle.LITERAL]
                cur_idx = styles.index(translator.translator.style)
                translator.translator.style = styles[(cur_idx + 1) % len(styles)]
                logger.info("Sentence style: %s", translator.translator.style.value)
            elif key == ord("d"):
                translator.draw_debug_skeleton = not translator.draw_debug_skeleton

    finally:
        cap.release()
        cv2.destroyAllWindows()
        translator.audio_engine.stop()
        logger.info("Live demonstration closed cleanly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time PSL Live Webcam Inference & Bilingual HUD")
    parser.add_argument("--webcam", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--video", type=str, default=None, help="Optional raw video file path to test instead of webcam")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained PyTorch model weights (.pth)")
    parser.add_argument("--flip", action="store_true", default=None, help="Force mirror/flip camera horizontally.")
    parser.add_argument("--no-flip", action="store_false", dest="flip", help="Disable camera horizontal flip.")
    args = parser.parse_args()

    run_live_webcam(
        webcam_index=args.webcam,
        video_source=args.video,
        checkpoint=args.checkpoint,
        flip_camera=args.flip,
    )
