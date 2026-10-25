"""Comprehensive Real-Time PSL Diagnostic & Skeletal Debugging Tool.

Academic Explainability:
    Provides deep inspection into the coordinate-space pipeline:
    1. Landmark Node Topology: Renders all 55 skeletal nodes with togglable index labels.
    2. Hand Centroid & Bounding Box Tracker: Isolates left and right hand clusters.
    3. Kinetic Energy & Motion Gating Telemetry: Live visualization of inter-frame joint velocities.
    4. Softmax Class Distribution: Real-time probability histogram and confidence margin inspection.
    5. Step-by-Step Frame Stepper: Enables microsecond-level debugging of dynamic sign transitions.
    6. Diagnostic Snapshot Tool: Dumps synchronized image, raw landmarks, and normalized tensors.
"""

import argparse
from collections import deque
import json
import logging
from pathlib import Path
import platform
import time
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from src.config import Config, VocabularyItem, load_config
from src.inference.audio_engine import AudioEngine
from src.inference.translation_layer import BilingualTranslator, SentenceStyle
from src.models.graph import SKELETAL_EDGES
from src.models.st_gcn import STGCN
from src.preprocessing.extract_landmarks import LandmarkExtractor
from src.preprocessing.normalize import center_coordinates, normalize_sequence, scale_normalize
from src.utils import get_device

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("PSLDebugger")


class PSLDiagnosticSession:
    """Diagnostic harness for inspecting ST-GCN landmark streams and gating dynamics."""

    def __init__(
        self,
        config: Optional[Config] = None,
        checkpoint_path: Optional[Path] = None,
    ) -> None:
        """Initializes the diagnostic session and loads underlying models."""
        self.config = config or load_config()
        self.device = get_device()
        logger.info("Initializing PSL Diagnostic Engine on: %s", self.device)

        self.extractor = LandmarkExtractor()
        self.window_size = self.config.data.sequence_length
        self.buffer: Deque[np.ndarray] = deque(maxlen=self.window_size)

        # Initialize model
        self.model = STGCN(
            in_channels=self.config.data.num_channels,
            num_classes=self.config.num_classes,
        ).to(self.device)
        self.model.eval()

        ckpt = checkpoint_path or (self.config.paths.checkpoints / "st_gcn_best.pth")
        if ckpt.exists():
            ckpt_data = torch.load(ckpt, map_location=self.device)
            state_dict = ckpt_data["model_state_dict"] if isinstance(ckpt_data, dict) and "model_state_dict" in ckpt_data else ckpt_data
            self.model.load_state_dict(state_dict)
            logger.info("Loaded ST-GCN weights from: %s", ckpt)
        else:
            logger.warning("No checkpoint found at %s. Running in unweighted test mode.", ckpt)

        # Thresholds
        self.motion_threshold = self.config.inference.motion_threshold
        self.min_hand_frames = self.config.inference.min_hand_frames
        self.conf_threshold = self.config.inference.confidence_threshold
        self.conf_margin = self.config.inference.confidence_margin

        # Diagnostic Flags
        self.show_node_labels: bool = False
        self.show_bounding_boxes: bool = True
        self.show_telemetry_hud: bool = True
        self.is_paused: bool = False
        self.step_frame: bool = False
        self.mirror_view: bool = True

        # Telemetry History
        self.session_events: List[Dict] = []
        self.frame_counter: int = 0
        self.active_sign_start: Optional[int] = None
        self.last_confirmed_sign: Optional[str] = None

    def analyze_frame(self, frame_bgr: np.ndarray) -> Dict:
        """Runs full feature extraction, kinetic motion energy calculation, and ST-GCN diagnosis.
        
        Args:
            frame_bgr: Raw camera/video frame.
            
        Returns:
            Dict: Comprehensive diagnostic telemetry dictionary.
        """
        self.frame_counter += 1
        t0 = time.perf_counter()

        # 1. Feature Extraction on raw optical frame
        landmarks = self.extractor.extract_frame_landmarks(frame_bgr)
        self.buffer.append(landmarks)
        t_extract = (time.perf_counter() - t0) * 1000.0

        # 2. Keypoint Group Validity
        lh_valid = bool(np.linalg.norm(landmarks[0:21]) > 1e-4)
        rh_valid = bool(np.linalg.norm(landmarks[21:42]) > 1e-4)
        pose_valid = bool(np.linalg.norm(landmarks[42:55]) > 1e-4)

        # 3. Buffer-wide kinetic motion energy
        buf_arr = np.array(self.buffer, dtype=np.float32)  # (F, 55, 3)
        hands = buf_arr[:, :42, :2]
        hand_frames_count = int(np.sum(np.any(np.linalg.norm(hands, axis=-1) > 1e-4, axis=1)))

        motion_energy = 0.0
        if len(buf_arr) >= 2:
            diffs = np.linalg.norm(np.diff(hands, axis=0), axis=-1)
            valid_pairs = (np.linalg.norm(hands[:-1], axis=-1) > 1e-4) & (np.linalg.norm(hands[1:], axis=-1) > 1e-4)
            valid_count = int(np.sum(valid_pairs))
            if valid_count > 0:
                motion_energy = float(np.sum(diffs * valid_pairs) / valid_count)

        # 4. Gating Verdict
        gate_buffer = len(self.buffer) == self.window_size
        gate_hands = hand_frames_count >= self.min_hand_frames
        gate_motion = motion_energy >= self.motion_threshold
        gate_open = gate_buffer and gate_hands and gate_motion

        # 5. Model Inference (if gate is open)
        top5 = []
        top1_class = "none"
        top1_conf = 0.0
        top2_conf = 0.0
        margin = 0.0
        t_infer = 0.0

        if gate_open:
            t_inf_start = time.perf_counter()
            norm_seq = normalize_sequence(buf_arr, target_length=self.window_size)
            tensor_batch = torch.from_numpy(norm_seq).permute(2, 0, 1).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(tensor_batch)
                probs = torch.softmax(logits, dim=-1)[0]
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)

                top1_idx = int(sorted_indices[0].item())
                top1_conf = float(sorted_probs[0].item())
                top2_conf = float(sorted_probs[1].item()) if len(sorted_probs) > 1 else 0.0
                margin = top1_conf - top2_conf
                top1_class = self.config.vocabulary[top1_idx].name

                top5 = [
                    (self.config.vocabulary[int(idx)].name, float(prob))
                    for prob, idx in zip(sorted_probs[:5], sorted_indices[:5])
                ]
            t_infer = (time.perf_counter() - t_inf_start) * 1000.0

        telemetry = {
            "frame_idx": self.frame_counter,
            "landmarks": landmarks,
            "lh_valid": lh_valid,
            "rh_valid": rh_valid,
            "pose_valid": pose_valid,
            "hand_frames": hand_frames_count,
            "motion_energy": motion_energy,
            "gate_buffer": gate_buffer,
            "gate_hands": gate_hands,
            "gate_motion": gate_motion,
            "gate_open": gate_open,
            "top1_class": top1_class,
            "top1_conf": top1_conf,
            "margin": margin,
            "top5": top5,
            "t_extract_ms": t_extract,
            "t_infer_ms": t_infer,
        }

        # Event tracking
        if gate_open and top1_conf >= self.conf_threshold and margin >= self.conf_margin:
            if self.last_confirmed_sign != top1_class:
                self.session_events.append({
                    "frame": self.frame_counter,
                    "event": "SIGN_DETECTED",
                    "sign": top1_class,
                    "confidence": top1_conf,
                    "margin": margin,
                    "motion": motion_energy,
                })
                self.last_confirmed_sign = top1_class
        elif not gate_open:
            self.last_confirmed_sign = None

        return telemetry

    def render_debug_frame(self, frame_bgr: np.ndarray, telemetry: Dict) -> np.ndarray:
        """Renders diagnostic overlay with high-contrast skeleton, node numbers, and telemetry."""
        h, w = frame_bgr.shape[:2]
        mirrored = self.mirror_view
        display = cv2.flip(frame_bgr, 1) if mirrored else frame_bgr.copy()
        landmarks = telemetry["landmarks"]

        def map_coord(pt: np.ndarray) -> Tuple[int, int]:
            disp_x = (1.0 - pt[0]) if mirrored else pt[0]
            return int(disp_x * w), int(pt[1] * h)

        # 1. Draw Bones
        for u, v in SKELETAL_EDGES:
            p1, p2 = landmarks[u], landmarks[v]
            if np.linalg.norm(p1) > 1e-4 and np.linalg.norm(p2) > 1e-4:
                x1, y1 = map_coord(p1)
                x2, y2 = map_coord(p2)
                col = (0, 255, 200) if u < 42 and v < 42 else (255, 160, 50)
                cv2.line(display, (x1, y1), (x2, y2), col, 2, cv2.LINE_AA)

        # 2. Draw Joints and optional Node Labels
        for i in range(55):
            pt = landmarks[i]
            if np.linalg.norm(pt) > 1e-4:
                x, y = map_coord(pt)
                if i < 21:
                    cv2.circle(display, (x, y), 3, (0, 255, 100), -1)  # Left hand
                elif i < 42:
                    cv2.circle(display, (x, y), 3, (0, 180, 255), -1)  # Right hand
                else:
                    cv2.circle(display, (x, y), 4, (255, 200, 0), -1)   # Pose

                if self.show_node_labels:
                    cv2.putText(display, str(i), (x + 4, y + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (255, 255, 255), 1)

        # 3. Draw Hand Bounding Boxes
        if self.show_bounding_boxes:
            for start_idx, end_idx, color, label in [
                (0, 21, (0, 255, 100), "Left Hand"),
                (21, 42, (0, 180, 255), "Right Hand"),
            ]:
                hand_pts = landmarks[start_idx:end_idx]
                valid = hand_pts[np.linalg.norm(hand_pts, axis=1) > 1e-4]
                if len(valid) > 0:
                    xs = [(1.0 - p[0]) * w if mirrored else p[0] * w for p in valid]
                    ys = [p[1] * h for p in valid]
                    min_x, max_x = max(0, int(min(xs)) - 10), min(w - 1, int(max(xs)) + 10)
                    min_y, max_y = max(0, int(min(ys)) - 10), min(h - 1, int(max(ys)) + 10)
                    cv2.rectangle(display, (min_x, min_y), (max_x, max_y), color, 1)
                    cv2.putText(display, label, (min_x, min_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA)

        # 4. Render Telemetry HUD Card
        if self.show_telemetry_hud:
            overlay = display.copy()
            hud_w = 420
            hud_h = 430
            hud_x = 20
            hud_y = 20
            cv2.rectangle(overlay, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (15, 18, 25), -1)
            gate_col = (0, 255, 150) if telemetry["gate_open"] else (80, 85, 100)
            cv2.rectangle(overlay, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), gate_col, 2)
            cv2.addWeighted(overlay, 0.88, display, 0.12, 0, display)

            # Header
            pause_tag = " [PAUSED]" if self.is_paused else ""
            cv2.putText(display, f"PSL DIAGNOSTIC TOOL v{self.config.version}{pause_tag}", (hud_x + 15, hud_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.line(display, (hud_x + 15, hud_y + 35), (hud_x + hud_w - 15, hud_y + 35), (50, 55, 70), 1)

            # Metric Rows
            rows = [
                f"Frame Number    : {telemetry['frame_idx']} (Extraction: {telemetry['t_extract_ms']:.1f}ms)",
                f"Hand Presence   : {telemetry['hand_frames']}/{self.window_size} frames in buffer",
                f"Hand Activity   : LH={'ACTIVE' if telemetry['lh_valid'] else 'NONE'} | RH={'ACTIVE' if telemetry['rh_valid'] else 'NONE'}",
                f"Kinetic Motion  : {telemetry['motion_energy']:.4f} (Threshold: {self.motion_threshold:.4f})",
                f"Gate Status     : {'OPEN (EVALUATING)' if telemetry['gate_open'] else 'CLOSED (IDLE/SUPPRESSED)'}",
            ]
            cur_y = hud_y + 55
            for r in rows:
                col = (0, 255, 150) if "OPEN" in r or "ACTIVE" in r else (200, 200, 210)
                cv2.putText(display, r, (hud_x + 15, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
                cur_y += 20

            cur_y += 5
            cv2.line(display, (hud_x + 15, cur_y), (hud_x + hud_w - 15, cur_y), (50, 55, 70), 1)
            cur_y += 20

            # Classification Candidates
            cv2.putText(display, "TOP CLASSIFICATION CANDIDATES:", (hud_x + 15, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)
            cur_y += 10

            if telemetry["top5"] and telemetry["gate_open"]:
                for rank, (name, prob) in enumerate(telemetry["top5"][:4], 1):
                    cur_y += 22
                    r_col = (0, 255, 180) if rank == 1 else (170, 175, 190)
                    cv2.putText(display, f"{rank}. {name:<14} : {prob * 100:5.1f}%", (hud_x + 15, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, r_col, 1, cv2.LINE_AA)

                    # mini bar
                    bar_w = 140
                    bx = hud_x + 240
                    cv2.rectangle(display, (bx, cur_y - 10), (bx + bar_w, cur_y - 2), (40, 45, 60), -1)
                    cv2.rectangle(display, (bx, cur_y - 10), (bx + int(bar_w * prob), cur_y - 2), r_col, -1)

                cur_y += 25
                margin_str = f"Margin: {telemetry['margin']*100:+.1f}% | Latency: {telemetry['t_infer_ms']:.1f}ms"
                cv2.putText(display, margin_str, (hud_x + 15, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 200, 0), 1, cv2.LINE_AA)
            else:
                cur_y += 35
                cv2.putText(display, "[EVALUATION GATED: Waiting for motion]", (hud_x + 20, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 125, 140), 1, cv2.LINE_AA)
                cur_y += 25

            # Hotkeys Help
            cur_y = hud_y + hud_h - 15
            cv2.putText(display, "[Space]: Pause | [N]: Nodes | [B]: Boxes | [S]: Save | [Q]: Quit", (hud_x + 15, cur_y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (140, 145, 160), 1, cv2.LINE_AA)

        return display

    def save_snapshot(self, frame_bgr: np.ndarray, telemetry: Dict) -> Path:
        """Dumps synchronized image, landmark array, and diagnostic JSON to logs/snapshots/."""
        snap_dir = self.config.paths.logs / "debug_snapshots"
        snap_dir.mkdir(parents=True, exist_ok=True)
        timestamp = int(time.time() * 1000)

        img_path = snap_dir / f"snapshot_{timestamp}_frame{telemetry['frame_idx']}.jpg"
        npy_path = snap_dir / f"snapshot_{timestamp}_landmarks.npy"
        json_path = snap_dir / f"snapshot_{timestamp}_telemetry.json"

        cv2.imwrite(str(img_path), frame_bgr)
        np.save(npy_path, telemetry["landmarks"])

        meta = {
            "timestamp": timestamp,
            "frame_idx": telemetry["frame_idx"],
            "lh_valid": telemetry["lh_valid"],
            "rh_valid": telemetry["rh_valid"],
            "motion_energy": telemetry["motion_energy"],
            "gate_open": telemetry["gate_open"],
            "top1_class": telemetry["top1_class"],
            "top1_conf": telemetry["top1_conf"],
            "margin": telemetry["margin"],
            "top5": telemetry["top5"],
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        logger.info("Saved diagnostic snapshot bundle to: %s", snap_dir)
        return json_path

    def print_session_summary(self) -> None:
        """Prints a comprehensive academic diagnostic summary report."""
        print("\n" + "=" * 80)
        print("  PSL TRANSLATOR -- DIAGNOSTIC SESSION SUMMARY REPORT")
        print("=" * 80)
        print(f"  Total Frames Evaluated : {self.frame_counter}")
        print(f"  Total Sign Detections  : {len(self.session_events)}")
        print("-" * 80)
        if self.session_events:
            print("  DETECTED SIGN EVENTS TIMELINE:")
            for ev in self.session_events:
                print(f"    Frame {ev['frame']:<5d} | Sign: {ev['sign']:<14s} | Conf: {ev['confidence']*100:5.1f}% | Margin: {ev['margin']*100:+5.1f}% | Motion: {ev['motion']:.3f}")
        else:
            print("  No signs met confidence & margin threshold during this session.")
        print("=" * 80 + "\n")


def run_debugger(
    video_path: Optional[str] = None,
    webcam_index: int = 0,
    checkpoint: Optional[str] = None,
) -> None:
    """Entry point for the interactive diagnostic tool."""
    session = PSLDiagnosticSession(checkpoint_path=Path(checkpoint) if checkpoint else None)
    source = video_path if video_path is not None else webcam_index
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        logger.error("Failed to open video source: %s", source)
        return

    session.mirror_view = (video_path is None)
    window_name = f"PSL Diagnostic Debugger - [{platform.system()}]"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("\n" + "=" * 80)
    print("  PSL DIAGNOSTIC TOOL INITIALIZED")
    print("=" * 80)
    print("  [Space] : Pause / Step Frame")
    print("  [N]     : Toggle Landmark Node IDs (0..54)")
    print("  [B]     : Toggle Hand Bounding Boxes")
    print("  [Tab]   : Toggle Telemetry HUD Overlay")
    print("  [F]     : Toggle Mirror View")
    print("  [S]     : Save Snapshot (Image + Landmarks + Telemetry JSON)")
    print("  [Q]     : Quit and Print Summary Report")
    print("=" * 80 + "\n")

    frame = None
    telemetry = None

    try:
        while True:
            if not session.is_paused or session.step_frame:
                ret, frame = cap.read()
                if not ret:
                    logger.info("End of video stream reached.")
                    break
                session.step_frame = False
                telemetry = session.analyze_frame(frame)

                # Terminal telemetry line
                gate_tag = "\033[92mOPEN\033[0m" if telemetry["gate_open"] else "\033[90mCLOSED\033[0m"
                top_str = f"{telemetry['top1_class']} ({telemetry['top1_conf']*100:.1f}%)" if telemetry["gate_open"] else "IDLE"
                sys.stdout.write(
                    f"\r[Frame {session.frame_counter:04d}] Motion: {telemetry['motion_energy']:.3f} [{gate_tag}] | "
                    f"Hands: {telemetry['hand_frames']:02d}/30 | Top: {top_str:<18s} | "
                    f"Margin: {telemetry['margin']*100:+5.1f}%"
                )
                sys.stdout.flush()

            if frame is not None and telemetry is not None:
                rendered = session.render_debug_frame(frame, telemetry)
                cv2.imshow(window_name, rendered)

            key = cv2.waitKey(1 if not session.is_paused else 30) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                session.is_paused = not session.is_paused
                logger.info("Diagnostic state: %s", "PAUSED" if session.is_paused else "RUNNING")
            elif key == ord(".") or key == ord(">"):
                if session.is_paused:
                    session.step_frame = True
            elif key == ord("n"):
                session.show_node_labels = not session.show_node_labels
            elif key == ord("b"):
                session.show_bounding_boxes = not session.show_bounding_boxes
            elif key == 9 or key == ord("\t"):
                session.show_telemetry_hud = not session.show_telemetry_hud
            elif key == ord("f"):
                session.mirror_view = not session.mirror_view
            elif key == ord("s"):
                if frame is not None and telemetry is not None:
                    session.save_snapshot(frame, telemetry)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        session.print_session_summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PSL Skeletal Diagnostic & Model Debugger")
    parser.add_argument("--video", type=str, default=None, help="Path to video file to diagnose")
    parser.add_argument("--webcam", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Custom checkpoint path")
    args = parser.parse_args()

    run_debugger(video_path=args.video, webcam_index=args.webcam, checkpoint=args.checkpoint)
