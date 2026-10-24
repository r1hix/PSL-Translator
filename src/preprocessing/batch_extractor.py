"""Batch landmark extraction and dataset processing orchestrator.

Processes all raw videos in `data/raw_videos/` through MediaPipe Holistic and
coordinate normalization, saving verified (30, 55, 3) float32 arrays into `data/landmarks/`.
"""

import argparse
import logging
from pathlib import Path
import time
from typing import Dict, List, Optional

import numpy as np

from src.config import Config, load_config
from src.preprocessing.extract_landmarks import LandmarkExtractor
from src.preprocessing.normalize import normalize_sequence

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("BatchExtractor")


def process_video_file(
    video_path: Path,
    output_path: Path,
    extractor: LandmarkExtractor,
    seq_length: int = 30,
    force_recompute: bool = False,
) -> bool:
    """Processes a single video into a normalized (30, 55, 3) landmark file.
    
    Args:
        video_path: Input video path.
        output_path: Destination .npy path.
        extractor: Initialized LandmarkExtractor instance.
        seq_length: Target frame length T (default: 30).
        force_recompute: Whether to overwrite existing .npy file.
        
    Returns:
        bool: True if successfully processed and saved.
    """
    if output_path.exists() and not force_recompute:
        return True

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        raw_seq = extractor.extract_video_landmarks(video_path, skip_empty_leads=True)
        if len(raw_seq) < 5:
            logger.warning("Video '%s' has insufficient valid frames (%d < 5). Skipping.", video_path.name, len(raw_seq))
            return False

        # Apply geometric normalization & temporal resampling -> (30, 55, 3)
        normalized_tensor = normalize_sequence(raw_seq, target_length=seq_length)

        # Sanity verification
        assert normalized_tensor.shape == (seq_length, 55, 3), f"Unexpected shape {normalized_tensor.shape}"
        assert not np.isnan(normalized_tensor).any(), "NaN detected in normalized tensor"
        assert not np.isinf(normalized_tensor).any(), "Inf detected in normalized tensor"

        np.save(output_path, normalized_tensor)
        logger.info("Saved: %s (Raw frames: %d -> Tensor: %s)", output_path.name, len(raw_seq), normalized_tensor.shape)
        return True

    except Exception as exc:
        logger.error("Failed to process '%s': %s", video_path.name, exc)
        return False


def run_batch_extraction(
    classes: Optional[List[str]] = None,
    force_recompute: bool = False,
    config_path: Optional[Path] = None,
) -> Dict[str, int]:
    """Orchestrates batch landmark extraction across all vocabulary classes.
    
    Args:
        classes: Optional subset of class names to process. Defaults to all in config.
        force_recompute: Overwrite already extracted .npy files.
        config_path: Path to YAML config.
        
    Returns:
        Dict[str, int]: Count of successful extractions per class.
    """
    config: Config = load_config(config_path)
    raw_dir = config.paths.raw_videos
    landmarks_dir = config.paths.landmarks
    seq_length = config.data.sequence_length

    target_classes = classes or [item.name for item in config.vocabulary]

    print("\n" + "=" * 80)
    print(f"[*] PSL BATCH LANDMARK EXTRACTION -- [T={seq_length}, V=55, C=3]")
    print(f"[*] Source Directory : {raw_dir}")
    print(f"[*] Target Directory : {landmarks_dir}")
    print("=" * 80)

    stats: Dict[str, int] = {}
    total_processed = 0
    total_found = 0
    start_time = time.time()

    with LandmarkExtractor() as extractor:
        for class_name in target_classes:
            class_raw_dir = raw_dir / class_name
            class_out_dir = landmarks_dir / class_name
            class_out_dir.mkdir(parents=True, exist_ok=True)

            if not class_raw_dir.exists():
                stats[class_name] = 0
                continue

            video_files = list(class_raw_dir.glob("*.mp4")) + list(class_raw_dir.glob("*.avi"))
            class_success = 0

            for v_file in video_files:
                total_found += 1
                out_file = class_out_dir / f"{v_file.stem}.npy"
                success = process_video_file(
                    v_file,
                    out_file,
                    extractor=extractor,
                    seq_length=seq_length,
                    force_recompute=force_recompute,
                )
                if success:
                    class_success += 1
                    total_processed += 1

            stats[class_name] = class_success
            print(f"  -> Class '{class_name:<16}': {class_success}/{len(video_files)} tensors extracted.")

    elapsed = time.time() - start_time
    print("-" * 80)
    print(f"Extraction Completed in {elapsed:.2f} seconds.")
    print(f"Total Tensors Generated: {total_processed}/{total_found}")
    print("=" * 80 + "\n")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract MediaPipe 55-keypoint tensors from raw videos.")
    parser.add_argument("--classes", type=str, default=None, help="Comma-separated class names to process.")
    parser.add_argument("--recompute", action="store_true", help="Force recomputation of existing .npy files.")
    args = parser.parse_args()

    selected = [c.strip() for c in args.classes.split(",")] if args.classes else None
    run_batch_extraction(classes=selected, force_recompute=args.recompute)
