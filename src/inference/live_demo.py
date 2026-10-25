"""Real-Time Pakistani Sign Language (PSL) Live Demonstration & Bilingual HUD.

Academic Explainability:
    Maintains a 30-frame rolling FIFO buffer (1.0 sec window at 30 FPS) of 55 skeletal
    joints extracted via MediaPipe Holistic. Keypoints are centered relative to the chest root
    and scale-normalized by inter-shoulder Euclidean distance before being dispatched to
    the ST-GCN neural model.

    To eliminate auditory chatter, false positives from resting postures, and closed-set
    softmax bias, the pipeline incorporates:
    1. Kinetic Motion Energy Gating: Suppresses evaluation when hand displacement is stationary (< threshold).
    2. Hand Presence Gating: Requires active hand keypoint detection in the rolling buffer.
    3. Confidence Margin Gating: Rejects predictions when Top-1 and Top-2 probabilities overlap (< margin).
    4. Optical Decoupling: Landmark extraction executes on raw unmirrored sensor coordinates, while
       user presentation supports seamless horizontal mirroring with synchronized overlay rendering.
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
    """End-to-End Real-Time Sign Language Inference, Gating, and HUD Interface."""

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
                ckpt_data = torch.load(ckpt, map_location=self.device)
                state_dict = ckpt_data["model_state_dict"] if isinstance(ckpt_data, dict) and "model_state_dict" in ckpt_data else ckpt_data
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

        # 4. Gating & Robustness Thresholds
        self.confidence_threshold: float = self.config.inference.confidence_threshold
        self.confidence_margin: float = self.config.inference.confidence_margin
        self.consecutive_trigger_frames: int = self.config.inference.consecutive_frames_trigger
        self.motion_threshold: float = self.config.inference.motion_threshold
        self.min_hand_frames: int = self.config.inference.min_hand_frames

        # 5. Runtime State Tracking
        self.recent_predictions: Deque[int] = deque(maxlen=self.consecutive_trigger_frames)
        self.current_motion_energy: float = 0.0
        self.current_hand_frames: int = 0
        self.is_lh_active: bool = False
        self.is_rh_active: bool = False
        self.gate_open: bool = False
        self.system_state_msg: str = "INITIALIZING"
        self.top5_predictions: List[Tuple[str, float]] = []
        self.current_margin: float = 0.0

        self.active_output: Optional[TranslatedOutput] = None
        self.active_confidence: float = 0.0
        self.last_detection_time: float = 0.0
        self.display_hold_sec: float = 3.2

        # 6. Presentation Settings
        self.draw_debug_skeleton: bool = self.config.inference.draw_landmarks
        self.mirror_display: bool = self.config.inference.mirror_display
        self.show_debug_panel: bool = self.config.inference.show_debug_panel
        self.urdu_font = get_urdu_font(size=36)
        self.urdu_font_small = get_urdu_font(size=22)

    def compute_motion_and_presence(self) -> Tuple[float, int, bool, bool]:
        """Calculates hand kinetic motion energy and hand detection count across the buffer.
        
        Academic Explainability:
            Computes mean inter-frame Euclidean displacement of valid hand keypoints (0..41).
            When a signer rests their hands or keeps them still, displacement is ~0.001 - 0.03,
            well below the dynamic threshold (0.10).
            
        Returns:
            Tuple[float, int, bool, bool]:
                - motion_energy: Mean Euclidean hand displacement per frame.
                - hand_frames: Number of frames in buffer with detected hand keypoints.
                - lh_active_now: Whether left hand is detected in the most recent frame.
                - rh_active_now: Whether right hand is detected in the most recent frame.
        """
        if not self.buffer:
            return 0.0, 0, False, False

        latest = self.buffer[-1]
        lh_active_now = bool(np.linalg.norm(latest[0:21]) > 1e-4)
        rh_active_now = bool(np.linalg.norm(latest[21:42]) > 1e-4)

        buf_arr = np.array(self.buffer, dtype=np.float32)  # (F, 55, 3)
        hands = buf_arr[:, :42, :2]  # (F, 42, 2)
        hand_frame_mask = np.any(np.linalg.norm(hands, axis=-1) > 1e-4, axis=1)
        hand_frames = int(np.sum(hand_frame_mask))

        if len(buf_arr) < 2:
            return 0.0, hand_frames, lh_active_now, rh_active_now

        # Inter-frame joint displacement
        diffs = np.linalg.norm(np.diff(hands, axis=0), axis=-1)  # (F-1, 42)
        valid_pairs = (np.linalg.norm(hands[:-1], axis=-1) > 1e-4) & (np.linalg.norm(hands[1:], axis=-1) > 1e-4)
        valid_count = int(np.sum(valid_pairs))

        if valid_count > 0:
            motion_energy = float(np.sum(diffs * valid_pairs) / valid_count)
        else:
            motion_energy = 0.0

        return motion_energy, hand_frames, lh_active_now, rh_active_now

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        mirror_display: Optional[bool] = None,
    ) -> Tuple[np.ndarray, Optional[TranslatedOutput]]:
        """Extracts landmarks on raw optical frame, applies motion gates, and renders HUD.
        
        Args:
            frame_bgr: Raw input camera frame of shape (H, W, 3).
            mirror_display: Optional override for screen mirroring.
            
        Returns:
            Tuple[np.ndarray, Optional[TranslatedOutput]]: Rendered frame and active prediction.
        """
        h, w = frame_bgr.shape[:2]
        should_mirror = self.mirror_display if mirror_display is None else mirror_display

        # 1. OPTICAL EXTRACTION: Always extract landmarks on the UNFLIPPED camera frame
        #    This preserves correct anatomical left/right hand assignment in MediaPipe.
        landmarks = self.extractor.extract_frame_landmarks(frame_bgr)
        self.buffer.append(landmarks)

        # 2. Compute kinetic motion energy and hand detection metrics
        motion, hand_frames, lh_active, rh_active = self.compute_motion_and_presence()
        self.current_motion_energy = motion
        self.current_hand_frames = hand_frames
        self.is_lh_active = lh_active
        self.is_rh_active = rh_active

        # 3. Create display frame (mirrored if requested for natural webcam experience)
        display_frame = cv2.flip(frame_bgr, 1) if should_mirror else frame_bgr.copy()

        # 4. Render Skeletal Topology Overlay if enabled
        if self.draw_debug_skeleton:
            self._draw_skeleton_overlay(display_frame, landmarks, w, h, mirrored=should_mirror)

        # 5. Gating Evaluation & Neural Model Dispatch
        is_buffer_full = len(self.buffer) == self.window_size
        has_hands = hand_frames >= self.min_hand_frames
        has_motion = motion >= self.motion_threshold

        if not is_buffer_full:
            self.gate_open = False
            self.system_state_msg = f"BUFFERING ({len(self.buffer)}/{self.window_size})"
        elif not has_hands:
            self.gate_open = False
            self.system_state_msg = "IDLE (NO HANDS DETECTED)"
            self.recent_predictions.clear()
        elif not has_motion:
            self.gate_open = False
            self.system_state_msg = "IDLE (HANDS RESTING / STATIONARY)"
            self.recent_predictions.clear()
        else:
            # GATE OPEN: User is actively signing with moving hands
            self.gate_open = True
            self.system_state_msg = "SIGNING DETECTED"

            raw_seq = np.array(self.buffer, dtype=np.float32)  # (30, 55, 3)
            norm_seq = normalize_sequence(raw_seq, target_length=self.window_size)
            tensor_batch = (
                torch.from_numpy(norm_seq)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(self.device)
            )

            with torch.no_grad():
                logits = self.model(tensor_batch)
                probabilities = torch.softmax(logits, dim=-1)[0]
                sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)

                top1_idx = int(sorted_indices[0].item())
                top1_conf = float(sorted_probs[0].item())
                top2_conf = float(sorted_probs[1].item()) if len(sorted_probs) > 1 else 0.0

                self.current_margin = top1_conf - top2_conf
                self.top5_predictions = [
                    (self.config.vocabulary[int(idx)].name, float(prob))
                    for prob, idx in zip(sorted_probs[:5], sorted_indices[:5])
                ]

            # Confidence and Margin checks to prevent overlapping class ambiguity
            is_confident = (top1_conf >= self.confidence_threshold) and (self.current_margin >= self.confidence_margin)

            if is_confident:
                self.recent_predictions.append(top1_idx)
                # Check consecutive frame stability
                is_consistent = (
                    len(self.recent_predictions) == self.consecutive_trigger_frames
                    and all(p == top1_idx for p in self.recent_predictions)
                )

                if is_consistent:
                    self.active_output = self.translator.translate(top1_idx, confidence=top1_conf)
                    self.active_confidence = top1_conf
                    self.last_detection_time = time.time()
                    self.system_state_msg = f"CONFIRMED: {self.active_output.word_name.upper()}"

                    # Asynchronous spoken Urdu pronunciation
                    self.audio_engine.play_word(
                        self.active_output.word_name,
                        communicative=(self.translator.style == SentenceStyle.COMMUNICATIVE),
                        async_mode=True,
                    )
            else:
                self.recent_predictions.clear()

        # 6. Auto-clear active prediction card after hold time when user returns to IDLE
        now = time.time()
        if self.active_output is not None and not self.gate_open:
            if (now - self.last_detection_time) > self.display_hold_sec:
                self.active_output = None

        # 7. Render Bilingual Glassmorphic HUD & Telemetry
        rendered_frame = self._render_hud(display_frame, w, h)
        if self.show_debug_panel:
            rendered_frame = self._render_debug_panel(rendered_frame, w, h)

        return rendered_frame, self.active_output

    def _draw_skeleton_overlay(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray,
        width: int,
        height: int,
        mirrored: bool = False,
    ) -> None:
        """Draws skeletal nodes and anatomical bones onto the OpenCV frame."""
        def map_x(norm_x: float) -> int:
            display_x = (1.0 - norm_x) if mirrored else norm_x
            return int(display_x * width)

        # 1. Draw bones
        for u, v in SKELETAL_EDGES:
            pt1 = landmarks[u]
            pt2 = landmarks[v]
            if np.linalg.norm(pt1) > 1e-4 and np.linalg.norm(pt2) > 1e-4:
                x1, y1 = map_x(pt1[0]), int(pt1[1] * height)
                x2, y2 = map_x(pt2[0]), int(pt2[1] * height)
                color = (0, 255, 200) if u < 42 and v < 42 else (255, 160, 50)
                cv2.line(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

        # 2. Draw joints
        for i in range(55):
            pt = landmarks[i]
            if np.linalg.norm(pt) > 1e-4:
                x, y = map_x(pt[0]), int(pt[1] * height)
                if i < 21:
                    cv2.circle(frame, (x, y), 3, (0, 255, 100), -1)   # Left hand (Green)
                elif i < 42:
                    cv2.circle(frame, (x, y), 3, (0, 180, 255), -1)   # Right hand (Orange)
                else:
                    cv2.circle(frame, (x, y), 4, (255, 200, 0), -1)   # Pose (Cyan/Yellow)

    def _render_hud(self, frame: np.ndarray, width: int, height: int) -> np.ndarray:
        """Composes a modern semi-transparent bilingual HUD banner using OpenCV and PIL."""
        overlay = frame.copy()

        # 1. Top Status Banner (Height: 50px)
        cv2.rectangle(overlay, (0, 0), (width, 50), (20, 20, 20), -1)

        # 2. Bottom Bilingual Translation Card (Height: 140px)
        card_h = 140
        card_y = height - card_h
        cv2.rectangle(overlay, (0, card_y), (width, height), (15, 17, 26), -1)
        border_color = (0, 230, 180) if self.gate_open else (70, 70, 85)
        cv2.line(overlay, (0, card_y), (width, card_y), border_color, 2)

        # Blend semi-transparent glassmorphism
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Top Bar Info
        dev_str = f"RIG: {self.device.type.upper()}"
        mode_str = f"MODE: {self.translator.mode.value.upper()}"
        audio_str = f"AUDIO: {'ON' if self.audio_engine.enabled else 'MUTED'}"
        gate_str = f"GATE: {'ACTIVE' if self.gate_open else 'IDLE'}"
        cv2.putText(
            frame,
            f"PSL-TRANSLATOR v{self.config.version} | {dev_str} | {mode_str} | {audio_str} | {gate_str}",
            (20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        buf_pct = min(100, int(len(self.buffer) / self.window_size * 100))
        cv2.putText(
            frame,
            f"Buffer: {buf_pct}%",
            (width - 130, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 255, 200) if buf_pct == 100 else (120, 120, 120),
            1,
            cv2.LINE_AA,
        )

        # Bottom Card Details
        if self.active_output is not None:
            # English Title / Roman Urdu
            title_text = f"SIGN: {self.active_output.word_name.upper()} ({self.active_output.roman_text}) -- Conf: {int(self.active_confidence * 100)}%"
            cv2.putText(frame, title_text, (30, card_y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 230, 180), 2, cv2.LINE_AA)

            # English Communicative Translation
            eng_sub = f"English: \"{self.active_output.english_text}\""
            cv2.putText(frame, eng_sub, (30, card_y + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

            # Urdu Native Script Rendering with PIL FreeType Font
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
            state_pill = f"STATE: {self.system_state_msg}"
            prompt_sub = "READY: Raise hand and gesture within camera frame to sign..."
            cv2.putText(frame, state_pill, (30, card_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 200, 255) if self.gate_open else (150, 150, 160), 2, cv2.LINE_AA)
            cv2.putText(frame, prompt_sub, (30, card_y + 85), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (200, 200, 200), 1, cv2.LINE_AA)

        return frame

    def _render_debug_panel(self, frame: np.ndarray, width: int, height: int) -> np.ndarray:
        """Renders an interactive real-time telemetry panel in the upper-right corner."""
        panel_w = 400
        panel_h = 420
        panel_x = width - panel_w - 20
        panel_y = 65

        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (18, 20, 30), -1)
        border_col = (0, 220, 180) if self.gate_open else (60, 65, 80)
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), border_col, 2)
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)

        # 1. Header
        status_dot = (0, 255, 120) if self.gate_open else (100, 100, 110)
        cv2.circle(frame, (panel_x + 20, panel_y + 24), 6, status_dot, -1)
        cv2.putText(frame, "LIVE TELEMETRY & GATING", (panel_x + 35, panel_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # 2. Motion Energy Meter
        curr_y = panel_y + 55
        motion_pct = min(1.0, self.current_motion_energy / (self.motion_threshold * 2.5 + 1e-6))
        gate_label = "GATE: OPEN" if self.gate_open else "GATE: CLOSED"
        gate_col = (0, 255, 120) if self.gate_open else (120, 120, 140)
        cv2.putText(frame, f"Motion: {self.current_motion_energy:.4f} (Min: {self.motion_threshold:.4f}) [{gate_label}]", (panel_x + 20, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, gate_col, 1, cv2.LINE_AA)

        curr_y += 10
        bar_w = panel_w - 40
        cv2.rectangle(frame, (panel_x + 20, curr_y), (panel_x + 20 + bar_w, curr_y + 8), (40, 45, 60), -1)
        cv2.rectangle(frame, (panel_x + 20, curr_y), (panel_x + 20 + int(bar_w * motion_pct), curr_y + 8), gate_col, -1)
        # Threshold marker
        thresh_x = panel_x + 20 + int(bar_w * (self.motion_threshold / (self.motion_threshold * 2.5)))
        cv2.line(frame, (thresh_x, curr_y - 2), (thresh_x, curr_y + 10), (0, 200, 255), 2)

        # 3. Hand Tracking Presence
        curr_y += 28
        lh_str = "LH: [ON]" if self.is_lh_active else "LH: [OFF]"
        rh_str = "RH: [ON]" if self.is_rh_active else "RH: [OFF]"
        lh_col = (0, 255, 100) if self.is_lh_active else (100, 100, 120)
        rh_col = (0, 180, 255) if self.is_rh_active else (100, 100, 120)
        cv2.putText(frame, f"Hand Buffer: {self.current_hand_frames}/{self.window_size} frames", (panel_x + 20, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(frame, lh_str, (panel_x + 230, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, lh_col, 1, cv2.LINE_AA)
        cv2.putText(frame, rh_str, (panel_x + 310, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, rh_col, 1, cv2.LINE_AA)

        # 4. Stability & Confidence Margin
        curr_y += 24
        stab_str = f"Stability: {len(self.recent_predictions)}/{self.consecutive_trigger_frames}"
        margin_str = f"Margin: {self.current_margin * 100:+.1f}%"
        cv2.putText(frame, f"{stab_str} | {margin_str}", (panel_x + 20, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 200), 1, cv2.LINE_AA)

        # 5. Top-5 Prediction Probabilities
        curr_y += 20
        cv2.line(frame, (panel_x + 20, curr_y), (panel_x + panel_w - 20, curr_y), (45, 50, 65), 1)
        curr_y += 20
        cv2.putText(frame, "TOP CLASSIFICATION CANDIDATES:", (panel_x + 20, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA)

        if self.top5_predictions and self.gate_open:
            for rank, (c_name, c_prob) in enumerate(self.top5_predictions, 1):
                curr_y += 28
                item = self.config.get_word_by_name(c_name)
                display_label = item.roman if item else c_name
                prob_pct = int(c_prob * 100)

                # Row Text
                rank_col = (0, 255, 180) if rank == 1 else (180, 180, 180)
                cv2.putText(frame, f"{rank}. {display_label[:14]:<14}", (panel_x + 20, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, rank_col, 1, cv2.LINE_AA)
                cv2.putText(frame, f"{prob_pct:>3d}%", (panel_x + 160, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, rank_col, 1, cv2.LINE_AA)

                # Mini Probability Bar
                pbar_x = panel_x + 205
                pbar_w = panel_w - 230
                cv2.rectangle(frame, (pbar_x, curr_y - 10), (pbar_x + pbar_w, curr_y), (40, 45, 60), -1)
                fill_w = int(pbar_w * c_prob)
                bar_fill_col = (0, 220, 180) if rank == 1 else (120, 130, 160)
                cv2.rectangle(frame, (pbar_x, curr_y - 10), (pbar_x + fill_w, curr_y), bar_fill_col, -1)
        else:
            curr_y += 40
            suppress_reason = "[SUPPRESSED: Hands Resting / Stationary]" if not self.gate_open else "[COLLECTING SEQUENCE...]"
            cv2.putText(frame, suppress_reason, (panel_x + 30, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 120, 130), 1, cv2.LINE_AA)

        # 6. Hotkey Footnote
        curr_y = panel_y + panel_h - 18
        cv2.putText(frame, "[Tab]: Hide Panel | [F]: Flip | [M]: Audio", (panel_x + 20, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 140, 160), 1, cv2.LINE_AA)

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

    # By default: mirror live webcam feed for natural user interaction; do not mirror pre-recorded videos
    should_mirror = (video_source is None) if flip_camera is None else flip_camera
    translator.mirror_display = should_mirror

    prev_time = time.time()
    fps = 0.0

    print("\n" + "=" * 80)
    print("  PSL REAL-TIME TRANSLATOR -- INTERACTIVE CONTROLS")
    print("=" * 80)
    print("  [Q]     : Quit live demonstration")
    print("  [Tab]   : Toggle Debug Telemetry Panel (Probabilities & Kinetic Energy)")
    print("  [F]     : Toggle Camera Horizontal Flip (Mirror Screen View)")
    print("  [M]     : Toggle Spoken Urdu Audio Mute / Unmute")
    print("  [L]     : Toggle Language Mode (Bilingual -> Urdu -> English)")
    print("  [S]     : Toggle Sentence Style (Communicative Sentence <-> Literal Word)")
    print("  [D]     : Toggle Landmark Skeletal Wireframe Overlay")
    print("  [C]     : Clear active sign display banner")
    print("=" * 80 + "\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("Video stream ended or frame unavailable.")
                break

            # Process frame through decoupled optical extraction, motion gating, and HUD
            rendered, _ = translator.process_frame(frame, mirror_display=should_mirror)

            # Compute and overlay FPS
            curr_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-4, curr_time - prev_time))
            prev_time = curr_time
            flip_tag = " [MIRROR]" if should_mirror else " [UNMIRRORED]"
            cv2.putText(
                rendered,
                f"FPS: {fps:.1f}{flip_tag}",
                (rendered.shape[1] - 280, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (0, 255, 120),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, rendered)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == 9 or key == ord("t"):  # [Tab] or [T]
                translator.show_debug_panel = not translator.show_debug_panel
                logger.info("Debug panel toggled: %s", "SHOWN" if translator.show_debug_panel else "HIDDEN")
            elif key == ord("f"):
                should_mirror = not should_mirror
                translator.mirror_display = should_mirror
                logger.info("Mirror display toggled: %s", "MIRROR" if should_mirror else "UNMIRRORED")
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
            elif key == ord("c"):
                translator.active_output = None
                translator.recent_predictions.clear()
                logger.info("Active sign banner cleared manually.")

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
