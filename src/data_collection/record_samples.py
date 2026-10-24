"""Interactive webcam recorder for capturing custom Pakistani Sign Language gesture clips.

Academic Explainability:
    Captures multi-subject dynamic sign sequences at fixed 30 FPS across 30 frames (1.0 sec),
    tagging files with subject IDs to facilitate strict subject-independent cross-validation.
"""

import argparse
from datetime import datetime
import logging
from pathlib import Path
import time
from typing import List, Optional

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    np = None
    CV2_AVAILABLE = False

from src.config import Config, VocabularyItem, load_config

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("SampleRecorder")


def draw_hud(
    frame: "np.ndarray",
    word_item: VocabularyItem,
    subject_id: str,
    status_text: str,
    frame_counter: int = 0,
    total_frames: int = 30,
) -> "np.ndarray":
    """Renders real-time guidance HUD on camera frame.
    
    Args:
        frame: BGR video frame array of shape (H, W, 3).
        word_item: Metadata for current sign being recorded.
        subject_id: Signer identifier (e.g. 'signer_01').
        status_text: Instructional status ('WAITING', 'COUNTDOWN', 'RECORDING').
        frame_counter: Current captured frame number.
        total_frames: Target sequence frame count (default: 30).
        
    Returns:
        np.ndarray: Frame with graphical HUD overlay.
    """
    canvas = frame.copy()
    h, w, _ = canvas.shape

    # Top overlay bar
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, 70), (20, 20, 20), -1)
    cv2.rectangle(overlay, (0, h - 50), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, canvas)

    # Top text: Target word info & subject
    title_text = f"Sign: {word_item.roman} ({word_item.name}) | Signer: {subject_id}"
    cv2.putText(canvas, title_text, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 230, 115), 2)
    cv2.putText(canvas, f"Urdu: {word_item.urdu}", (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Status banner in center if waiting or countdown
    if "COUNTDOWN" in status_text:
        cv2.putText(canvas, status_text, (w // 2 - 120, h // 2), cv2.FONT_HERSHEY_DUPLEX, 1.6, (0, 215, 255), 3)
    elif "RECORDING" in status_text:
        # Red recording dot and progress bar
        cv2.circle(canvas, (w - 40, 35), 12, (0, 0, 255), -1)
        cv2.putText(canvas, f"REC [{frame_counter}/{total_frames}]", (w - 180, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Progress bar at bottom
        progress_w = int((frame_counter / total_frames) * (w - 40))
        cv2.rectangle(canvas, (20, h - 20), (20 + progress_w, h - 10), (0, 230, 115), -1)
    else:
        # Bottom controls helper
        controls = "Controls: [SPACE] Record | [R] Redo | [N] Next Sign | [Q] Quit"
        cv2.putText(canvas, controls, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)

    return canvas


def record_word_samples(
    word_name: str,
    subject_id: str,
    num_samples: int = 5,
    camera_index: int = 0,
    config_path: Optional[Path] = None,
) -> int:
    """Records multiple 30-frame video samples for a specific word.
    
    Args:
        word_name: Canonical English name of sign.
        subject_id: Signer/subject identifier.
        num_samples: Number of video takes to collect.
        camera_index: OpenCV VideoCapture device index.
        config_path: Optional Path to config.yaml.
        
    Returns:
        int: Total valid clips recorded and saved.
    """
    if not CV2_AVAILABLE:
        logger.error("OpenCV (cv2) is not installed. Please install opencv-python.")
        return 0

    config: Config = load_config(config_path)
    word_item = config.get_word_by_name(word_name)
    if not word_item:
        logger.error("Word '%s' not found in config vocabulary.", word_name)
        return 0

    save_dir = config.paths.raw_videos / word_name
    save_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logger.error("Unable to open camera device %d.", camera_index)
        return 0

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.raw_dict["inference"].get("frame_width", 1280))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.raw_dict["inference"].get("frame_height", 720))
    cap.set(cv2.CAP_PROP_FPS, 30)

    fps = 30.0
    seq_len = config.data.sequence_length  # Exactly 30 frames
    completed = 0

    logger.info("Ready to record '%s' (%s). Target: %d samples.", word_item.roman, word_item.urdu, num_samples)

    try:
        while completed < num_samples:
            # Phase 1: Wait for user to trigger recording take
            waiting = True
            while waiting:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)  # Mirror view
                status = f"Sample {completed + 1}/{num_samples} - Press SPACE to begin"
                hud = draw_hud(frame, word_item, subject_id, status)
                cv2.imshow("PSL Sample Recorder", hud)

                key = cv2.waitKey(1) & 0xFF
                if key == ord(" "):
                    waiting = False
                elif key == ord("q"):
                    logger.info("Recording session cancelled by user.")
                    return completed

            # Phase 2: 3-Second Countdown
            for sec in range(3, 0, -1):
                start_time = time.time()
                while time.time() - start_time < 1.0:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.flip(frame, 1)
                    hud = draw_hud(frame, word_item, subject_id, f"COUNTDOWN: {sec}")
                    cv2.imshow("PSL Sample Recorder", hud)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        return completed

            # Phase 3: Record exactly 30 frames
            captured_frames: List["np.ndarray"] = []
            logger.info("Recording sample %d/%d...", completed + 1, num_samples)

            for f_idx in range(1, seq_len + 1):
                frame_start = time.time()
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                captured_frames.append(frame.copy())

                hud = draw_hud(frame, word_item, subject_id, "RECORDING", f_idx, seq_len)
                cv2.imshow("PSL Sample Recorder", hud)
                cv2.waitKey(1)

                # Maintain 30 FPS timing
                elapsed = time.time() - frame_start
                if elapsed < (1.0 / fps):
                    time.sleep((1.0 / fps) - elapsed)

            # Phase 4: Save video take
            if len(captured_frames) == seq_len:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{subject_id}_{word_name}_{timestamp}.mp4"
                file_path = save_dir / filename

                h, w, _ = captured_frames[0].shape
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(file_path), fourcc, fps, (w, h))

                for f in captured_frames:
                    writer.write(f)
                writer.release()

                completed += 1
                logger.info("Saved sample to: %s", file_path.name)
            else:
                logger.warning("Frame count mismatch (%d/%d). Discarded take.", len(captured_frames), seq_len)

    finally:
        cap.release()
        cv2.destroyAllWindows()

    logger.info("Completed %d/%d recordings for '%s'.", completed, num_samples, word_name)
    return completed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record 30-frame webcam samples for PSL sign words.")
    parser.add_argument("--word", type=str, required=True, help="Canonical word name (e.g. 'pani', 'doctor').")
    parser.add_argument("--subject_id", type=str, default="signer_01", help="Signer/Subject identifier.")
    parser.add_argument("--samples", type=int, default=5, help="Number of 30-frame samples to record.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index.")
    args = parser.parse_args()

    record_word_samples(args.word, args.subject_id, args.samples, args.camera)
