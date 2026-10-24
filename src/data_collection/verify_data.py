"""Data verification and integrity audit utility for raw PSL video datasets.

Scans `data/raw_videos/`, checks container readability, dimensions, framerate,
and generates class-by-class inventory reports for academic logging.
"""

import argparse
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Dict, List, Optional

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    CV2_AVAILABLE = False

from src.config import Config, load_config

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("DataVerifier")


@dataclass
class VideoMetadata:
    """Stores inspected characteristics of a raw video file."""
    path: Path
    class_name: str
    frame_count: int
    fps: float
    width: int
    height: int
    duration_sec: float
    is_valid: bool
    error_msg: str = ""


def inspect_video(video_path: Path, class_name: str, expected_seq_len: int = 30) -> VideoMetadata:
    """Inspects video container and frames using OpenCV.
    
    Args:
        video_path: Path to .mp4 video file.
        class_name: Canonical class label.
        expected_seq_len: Target sequence length in frames.
        
    Returns:
        VideoMetadata: Extracted video metrics and validity status.
    """
    if not CV2_AVAILABLE:
        return VideoMetadata(video_path, class_name, 0, 0.0, 0, 0, 0.0, False, "OpenCV not installed")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return VideoMetadata(video_path, class_name, 0, 0.0, 0, 0, 0.0, False, "Could not open video file")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    duration = frame_count / fps if fps > 0 else 0.0
    
    # A video is considered valid if it has at least 15 frames (which can be interpolated/sampled to 30)
    # and has valid spatial dimensions.
    is_valid = frame_count >= 15 and width > 0 and height > 0
    error_msg = "" if is_valid else f"Insufficient frames ({frame_count} < 15) or invalid dimensions ({width}x{height})"

    return VideoMetadata(
        path=video_path,
        class_name=class_name,
        frame_count=frame_count,
        fps=fps,
        width=width,
        height=height,
        duration_sec=duration,
        is_valid=is_valid,
        error_msg=error_msg,
    )


def audit_raw_dataset(config_path: Optional[Path] = None) -> Dict[str, List[VideoMetadata]]:
    """Audits entire `data/raw_videos` directory across all vocabulary classes.
    
    Args:
        config_path: Optional explicit Path to config.yaml.
        
    Returns:
        Dict[str, List[VideoMetadata]]: Grouped metadata by class label.
    """
    config: Config = load_config(config_path)
    raw_dir = config.paths.raw_videos
    expected_len = config.data.sequence_length

    results: Dict[str, List[VideoMetadata]] = {item.name: [] for item in config.vocabulary}
    total_videos = 0
    valid_videos = 0

    print("\n" + "=" * 80)
    print(f"[*] PSL RAW VIDEO DATASET AUDIT -- [{config.project_name} v{config.version}]")
    print("=" * 80)

    print(f"{'Class Name':<16} | {'Samples':<8} | {'Valid':<8} | {'Avg Frames':<12} | {'Avg Duration':<12}")
    print("-" * 80)

    for item in config.vocabulary:
        class_folder = raw_dir / item.name
        if not class_folder.exists():
            print(f"{item.name:<16} | {'0':<8} | {'0':<8} | {'N/A':<12} | {'N/A':<12}")
            continue

        video_files = list(class_folder.glob("*.mp4")) + list(class_folder.glob("*.avi"))
        class_meta: List[VideoMetadata] = []

        for v_path in video_files:
            meta = inspect_video(v_path, item.name, expected_len)
            class_meta.append(meta)
            total_videos += 1
            if meta.is_valid:
                valid_videos += 1

        results[item.name] = class_meta

        sample_count = len(class_meta)
        class_valid_count = sum(1 for m in class_meta if m.is_valid)
        avg_frames = (
            f"{sum(m.frame_count for m in class_meta) / sample_count:.1f}"
            if sample_count > 0
            else "N/A"
        )
        avg_dur = (
            f"{sum(m.duration_sec for m in class_meta) / sample_count:.2f}s"
            if sample_count > 0
            else "N/A"
        )

        print(f"{item.name:<16} | {sample_count:<8} | {class_valid_count:<8} | {avg_frames:<12} | {avg_dur:<12}")

    print("-" * 80)
    print(f"Total Video Files Found : {total_videos}")
    print(f"Total Valid Clips       : {valid_videos}")
    print(f"Inventory Health Ratio  : {(valid_videos / total_videos * 100):.1f}%" if total_videos > 0 else "Inventory Health Ratio  : 0.0% (No videos found yet)")
    print("=" * 80 + "\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify raw PSL video file integrity and sample tally.")
    parser.parse_args()
    audit_raw_dataset()
