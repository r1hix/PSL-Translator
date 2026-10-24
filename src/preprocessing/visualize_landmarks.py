"""Interactive visualizer for 55-keypoint skeletal landmarks and motion graphs.

Academic Explainability:
    Renders anatomical bone connections for the 55-node skeletal graph:
    - Left Hand (cyan): 21 nodes (indices 0 to 20)
    - Right Hand (magenta): 21 nodes (indices 21 to 41)
    - Upper Body Pose (lime green): 13 nodes (indices 42 to 54)
    - Wrist-to-Hand Bridge: Connects pose wrists (51, 52) to hand bases (0, 21)
"""

import argparse
import logging
from pathlib import Path
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.config import Config, load_config
from src.preprocessing.extract_landmarks import LandmarkExtractor
from src.preprocessing.normalize import center_coordinates, normalize_sequence, scale_normalize

logger = logging.getLogger("Visualizer")

# ------------------------------------------------------------------------------
# Anatomical Bone Connectivity (Edges E in Skeletal Graph G = (V, E))
# ------------------------------------------------------------------------------
# Intra-Hand Bones for 21 joints (relative to wrist 0)
HAND_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),# Ring
    (0, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (5, 9), (9, 13), (13, 17),            # Knuckle palm arc
]

# Left Hand Edges (0 to 20)
LEFT_HAND_EDGES = [(u, v) for u, v in HAND_BONES]

# Right Hand Edges (21 to 41)
RIGHT_HAND_EDGES = [(u + 21, v + 21) for u, v in HAND_BONES]

# Upper Body Pose Edges (indices 42 to 54)
# 42:Nose, 43:LEye, 44:REye, 45:LEar, 46:REar,
# 47:LShoulder, 48:RShoulder, 49:LElbow, 50:RElbow,
# 51:LWrist, 52:RWrist, 53:LHip, 54:RHip
POSE_EDGES = [
    (42, 43), (43, 45),                   # Left face
    (42, 44), (44, 46),                   # Right face
    (47, 48),                             # Shoulder span
    (47, 49), (49, 51),                   # Left arm
    (48, 50), (50, 52),                   # Right arm
    (47, 53), (48, 54), (53, 54),         # Torso frame
    (51, 0),                              # Bridge: Left Pose Wrist -> Left Hand Root
    (52, 21),                             # Bridge: Right Pose Wrist -> Right Hand Root
]

ALL_EDGES = LEFT_HAND_EDGES + RIGHT_HAND_EDGES + POSE_EDGES


def draw_skeleton_on_canvas(
    canvas: np.ndarray,
    landmarks_55: np.ndarray,
    is_normalized_space: bool = False,
) -> np.ndarray:
    """Draws colored skeletal joints and connecting bones on an image canvas.
    
    Args:
        canvas: Image array of shape (H, W, 3).
        landmarks_55: Array of shape (55, 3) with (x, y, z) coordinates.
        is_normalized_space: If True, coords are centered at (0, 0) and scaled around 0.
        
    Returns:
        np.ndarray: Canvas with rendered skeletal graph.
    """
    h, w, _ = canvas.shape

    # Convert coordinates to pixel space (px, py)
    pts = np.zeros((55, 2), dtype=int)
    valid_mask = np.zeros(55, dtype=bool)

    for i in range(55):
        x, y = landmarks_55[i, 0], landmarks_55[i, 1]
        if np.abs(x) > 1e-5 or np.abs(y) > 1e-5:
            valid_mask[i] = True
            if is_normalized_space:
                # Map centered normalized space [-1.5, 1.5] to canvas center
                px = int(w / 2 + x * (w * 0.35))
                py = int(h / 2 + y * (h * 0.35))
            else:
                # Standard relative coords [0, 1]
                px = int(x * w)
                py = int(y * h)
            pts[i] = [np.clip(px, 0, w - 1), np.clip(py, 0, h - 1)]

    # Draw Bones
    for u, v in ALL_EDGES:
        if valid_mask[u] and valid_mask[v]:
            pt1 = tuple(pts[u])
            pt2 = tuple(pts[v])

            if u < 21 and v < 21:
                color = (255, 255, 0)  # Cyan: Left Hand
                thickness = 2
            elif 21 <= u < 42 and 21 <= v < 42:
                color = (255, 0, 255)  # Magenta: Right Hand
                thickness = 2
            elif (u == 51 and v == 0) or (u == 52 and v == 21):
                color = (200, 200, 200) # Gray: Wrist-to-Hand Bridge
                thickness = 1
            else:
                color = (0, 255, 128)  # Lime Green: Pose Body
                thickness = 3

            cv2.line(canvas, pt1, pt2, color, thickness, cv2.LINE_AA)

    # Draw Joints
    for i in range(55):
        if valid_mask[i]:
            center = tuple(pts[i])
            if i < 21:
                j_color = (0, 255, 255)  # Yellow dots: Left hand
                radius = 3
            elif i < 42:
                j_color = (255, 128, 255) # Pink dots: Right hand
                radius = 3
            else:
                j_color = (0, 255, 0)    # Green dots: Pose
                radius = 5
            cv2.circle(canvas, center, radius, j_color, -1, cv2.LINE_AA)

    return canvas


def visualize_video(
    video_path: Path,
    extractor: LandmarkExtractor,
    overlay_mode: bool = False,
) -> None:
    """Plays video with real-time MediaPipe 55-keypoint graph visualization.
    
    Args:
        video_path: Video file to inspect.
        extractor: Initialized LandmarkExtractor instance.
        overlay_mode: Whether to overlay directly or show side-by-side.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Unable to open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    delay = int(1000 / fps)

    print(f"\n[*] Playing: {video_path.name}")
    print("[*] Controls: [SPACE] Pause/Resume | [O] Toggle Overlay/Side-by-Side | [Q] Exit\n")

    paused = False
    mode_overlay = overlay_mode

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop playback
                continue

            # Standardize resolution
            frame = cv2.resize(frame, (854, 480))
            h, w, _ = frame.shape

            # Extract 55 landmarks
            landmarks = extractor.extract_frame_landmarks(frame)

            # Create black skeletal canvas
            black_canvas = np.zeros_like(frame)
            draw_skeleton_on_canvas(black_canvas, landmarks)

            if mode_overlay:
                display = cv2.addWeighted(frame, 0.7, black_canvas, 1.0, 0)
            else:
                # Side-by-side view
                draw_skeleton_on_canvas(frame, landmarks)
                display = np.hstack([frame, black_canvas])

            # Info text
            cv2.putText(
                display,
                f"File: {video_path.stem} | 55 Joints (Cyan: L-Hand, Magenta: R-Hand, Green: Pose)",
                (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 230, 115),
                2,
            )

        cv2.imshow("PSL 55-Joint Skeletal Visualizer", display)
        key = cv2.waitKey(delay) & 0xFF

        if key == ord("q"):
            break
        elif key == ord(" "):
            paused = not paused
        elif key == ord("o"):
            mode_overlay = not mode_overlay

    cap.release()
    cv2.destroyAllWindows()


def visualize_webcam(extractor: LandmarkExtractor) -> None:
    """Streams live webcam feed with real-time 55-keypoint graph rendering.
    
    Args:
        extractor: Initialized LandmarkExtractor instance.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access default webcam device (index 0).")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n[*] Streaming live webcam...")
    print("[*] Controls: [O] Toggle Overlay | [Q] Quit\n")

    mode_overlay = True
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Mirror view
        h, w, _ = frame.shape

        now = time.time()
        fps_val = 1.0 / (now - prev_time) if now > prev_time else 30.0
        prev_time = now

        landmarks = extractor.extract_frame_landmarks(frame)

        black_canvas = np.zeros_like(frame)
        draw_skeleton_on_canvas(black_canvas, landmarks)

        if mode_overlay:
            display = cv2.addWeighted(frame, 0.75, black_canvas, 1.0, 0)
        else:
            display = np.hstack([frame, black_canvas])

        cv2.putText(display, f"Live FPS: {fps_val:.1f} | 55 Nodes (Press Q to exit)", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 128), 2)
        cv2.imshow("PSL Live Webcam Tracker", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("o"):
            mode_overlay = not mode_overlay

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize MediaPipe 55-keypoint skeletal tracking.")
    parser.add_argument("--video", type=str, default=None, help="Path to video file.")
    parser.add_argument("--webcam", action="store_true", help="Stream from live webcam.")
    parser.add_argument("--class_name", type=str, default="doctor", help="Inspect first video of a class name.")
    args = parser.parse_args()

    with LandmarkExtractor() as ext:
        if args.webcam:
            visualize_webcam(ext)
        else:
            if args.video:
                v_path = Path(args.video)
            else:
                config = load_config()
                v_dir = config.paths.raw_videos / args.class_name
                videos = list(v_dir.glob("*.mp4"))
                v_path = videos[0] if videos else None

            if v_path and v_path.exists():
                visualize_video(v_path, ext)
            else:
                print(f"No video found for class '{args.class_name}'. Pass --video <path> or --webcam.")
