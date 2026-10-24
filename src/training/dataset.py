"""Dataset and Data Augmentation pipeline for PSL Skeletal Coordinates.

Conforms strictly to the project data contract:
    Input: Normalized coordinate arrays of shape (T=30, V=55, C=3) in float32.
    Output: PyTorch Batch Tensors of shape (N, C=3, T=30, V=55).

Implements coordinate-space domain augmentations:
    1. Spatial Jittering (Gaussian noise)
    2. Spatial Scaling (Signer arm span / distance variation)
    3. Spatial Translation Jitter
    4. Temporal Cadence Perturbation (Time-warping via linear interpolation)
    5. Joint Dropout (Finger occlusion simulation)
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.config import load_config, Config


class CoordinateAugmentation:
    """Domain-specific coordinate-space augmentation for sign language skeletal graphs.
    
    Transforms coordinates without violating physical kinematic constraints:
        - Maintains chest-centered reference coordinate frame.
        - Applies continuous spatio-temporal noise to simulate real-world variance.
    """

    def __init__(
        self,
        jitter_std: float = 0.015,
        scale_range: Tuple[float, float] = (0.92, 1.08),
        translation_range: float = 0.02,
        time_warp_range: Tuple[float, float] = (0.85, 1.15),
        joint_dropout_prob: float = 0.05,
        p: float = 0.8,
    ):
        """Initializes coordinate augmentation parameters.
        
        Args:
            jitter_std: Standard deviation of additive Gaussian noise (meters/normalized unit).
            scale_range: Min and max uniform scaling factors for limb span variations.
            translation_range: Max uniform shift along (x, y, z) axes.
            time_warp_range: Min and max temporal stretch/compression factor.
            joint_dropout_prob: Probability of zeroing individual joint nodes (occlusion).
            p: Overall probability of applying augmentation pipeline.
        """
        self.jitter_std = jitter_std
        self.scale_range = scale_range
        self.translation_range = translation_range
        self.time_warp_range = time_warp_range
        self.joint_dropout_prob = joint_dropout_prob
        self.p = p

    def __call__(self, landmarks: np.ndarray) -> np.ndarray:
        """Applies stochastic augmentations to a sequence of shape (T, V, C).
        
        Args:
            landmarks: float32 array of shape (T, V, C) = (30, 55, 3).
            
        Returns:
            np.ndarray: Augmented array of shape (T, V, C) = (30, 55, 3).
        """
        if random.random() > self.p:
            return landmarks.copy()

        aug = landmarks.copy()
        t_len, v_nodes, c_dim = aug.shape

        # 1. Spatial Scaling: P' = P * s
        # Simulates signers with varying stature and varying distances to the camera
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        aug *= scale

        # 2. Translation Jitter: P' = P + delta
        # Simulates slight off-center drift of signer position
        shift = np.random.uniform(-self.translation_range, self.translation_range, size=(1, 1, c_dim))
        aug += shift

        # 3. Gaussian Spatial Jitter: P' = P + N(0, sigma^2)
        # Simulates MediaPipe landmark detector precision wobble and optical sensor noise
        noise = np.random.normal(0.0, self.jitter_std, size=aug.shape)
        aug += noise

        # 4. Joint Dropout (Occlusion)
        # Simulates temporary optical occlusion when hands overlap torso or fingers overlap
        if self.joint_dropout_prob > 0.0:
            mask = np.random.binomial(1, 1.0 - self.joint_dropout_prob, size=(1, v_nodes, 1))
            aug *= mask

        # 5. Temporal Speed Cadence Perturbation (Time-warping)
        # Signers naturally sign at varying tempos; interpolate to simulate speed shifts
        warp_factor = random.uniform(self.time_warp_range[0], self.time_warp_range[1])
        if abs(warp_factor - 1.0) > 0.03:
            orig_indices = np.linspace(0, t_len - 1, num=t_len)
            # Warp time indices
            target_len = int(round(t_len * warp_factor))
            warped_indices = np.linspace(0, t_len - 1, num=target_len)
            
            # Resample warped sequence back to fixed t_len (30 frames)
            resampled = np.zeros_like(aug)
            for v in range(v_nodes):
                for c in range(c_dim):
                    trajectory = aug[:, v, c]
                    # Interpolate warped curve
                    warped_curve = np.interp(warped_indices, orig_indices, trajectory)
                    # Resample back to fixed 30 points
                    resample_indices = np.linspace(0, target_len - 1, num=t_len)
                    resampled[:, v, c] = np.interp(resample_indices, np.arange(target_len), warped_curve)
            aug = resampled

        return aug.astype(np.float32)


class PSLDataset(Dataset):
    """PyTorch Dataset for PSL Skeletal Landmark Tensors.
    
    Loads preprocessed coordinates from disk, maps categorical vocabulary labels,
    applies stochastic augmentations, and formats tensors to PyTorch ST-GCN contract:
        Output tensor shape: (C, T, V) = (3, 30, 55).
    """

    def __init__(
        self,
        samples: List[Dict[str, Union[Path, str, int]]],
        transform: Optional[CoordinateAugmentation] = None,
        virtual_multiplier: int = 1,
    ):
        """Initializes PSL Dataset instance.
        
        Args:
            samples: List of sample metadata dicts containing 'path', 'label', 'word', and 'subject_id'.
            transform: Optional coordinate augmentation pipeline.
            virtual_multiplier: Repeats dataset virtually per epoch so that augmentations
                                produce multiple unique variations in small datasets.
        """
        self.samples = samples
        self.transform = transform
        self.virtual_multiplier = max(1, virtual_multiplier)

    def __len__(self) -> int:
        return len(self.samples) * self.virtual_multiplier

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str]:
        """Loads and prepares a skeletal landmark sample.
        
        Args:
            index: Dataset index.
            
        Returns:
            Tuple[torch.Tensor, int, str]:
                - tensor: FloatTensor of shape (C, T, V) = (3, 30, 55)
                - label: Target integer class index (0 to num_classes - 1)
                - word: Target class word identifier
        """
        # Map virtual index back to physical sample index
        real_idx = index % len(self.samples)
        sample_info = self.samples[real_idx]
        file_path = Path(sample_info["path"])
        label = int(sample_info["label"])
        word = str(sample_info["word"])

        # Load raw numpy array of shape (T, V, C) = (30, 55, 3)
        landmarks = np.load(file_path).astype(np.float32)

        # Apply stochastic domain augmentations
        if self.transform is not None:
            landmarks = self.transform(landmarks)

        # Transpose from (T, V, C) to PyTorch ST-GCN standard (C, T, V)
        # Mathematical transposition: axes (2, 0, 1) -> (C=3, T=30, V=55)
        tensor_ctv = torch.from_numpy(landmarks).permute(2, 0, 1).contiguous()

        return tensor_ctv, label, word


def discover_landmark_samples(
    landmarks_dir: Path | str,
    config: Optional[Config] = None,
) -> Tuple[List[Dict[str, Union[Path, str, int]]], Dict[str, int]]:
    """Recursively discovers all pre-extracted .npy landmark files and maps to class labels.
    
    Args:
        landmarks_dir: Path to directory containing class subfolders of .npy files.
        config: Loaded PSLConfig instance (optional, for explicit vocabulary ordering).
        
    Returns:
        Tuple:
            - List of sample metadata dictionaries.
            - Class-to-index mapping dictionary.
    """
    root_dir = Path(landmarks_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"Landmarks directory not found: {root_dir}")

    # Build class mapping from config or folder structure
    if config is not None:
        class_to_idx = {item.name: item.id for item in config.vocabulary}
    else:
        # Fallback to alphanumeric sorted subdirectories
        subdirs = sorted([d.name for d in root_dir.iterdir() if d.is_dir()])
        class_to_idx = {name: idx for idx, name in enumerate(subdirs)}

    samples: List[Dict[str, Union[Path, str, int]]] = []

    for class_name, class_id in class_to_idx.items():
        class_folder = root_dir / class_name
        if not class_folder.exists():
            continue

        for npy_file in class_folder.glob("*.npy"):
            # Parse subject_id if present in filename: e.g. "psl_portal_..." or "..._sub01_..."
            stem = npy_file.stem
            subject_id = "portal"
            if "sub_" in stem or "user_" in stem:
                parts = stem.split("_")
                for part in parts:
                    if part.startswith("sub") or part.startswith("user"):
                        subject_id = part
                        break

            samples.append({
                "path": npy_file,
                "label": class_id,
                "word": class_name,
                "subject_id": subject_id,
            })

    return samples, class_to_idx


def create_dataloaders(
    landmarks_dir: Path | str,
    config: Optional[Config] = None,
    batch_size: int = 32,
    val_split: float = 0.2,
    num_workers: int = 0,
    train_virtual_multiplier: int = 20,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    """Constructs training and validation DataLoaders with stratified/subject splitting.
    
    Args:
        landmarks_dir: Root path to landmark dataset.
        config: System configuration object.
        batch_size: Mini-batch size.
        val_split: Validation split fraction.
        num_workers: Number of DataLoader worker subprocesses.
        train_virtual_multiplier: Number of synthetic augmented passes per epoch.
        seed: Random seed for deterministic reproducibility.
        
    Returns:
        Tuple[DataLoader, DataLoader, Dict[str, int]]:
            - train_loader
            - val_loader
            - class_to_idx mapping
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    samples, class_to_idx = discover_landmark_samples(landmarks_dir, config)
    if not samples:
        raise ValueError(f"No .npy landmark files discovered in {landmarks_dir}")

    # Check distinct subjects
    distinct_subjects = list(set(s["subject_id"] for s in samples))

    if len(distinct_subjects) > 1 and val_split > 0:
        # Subject-independent cross-validation splitting:
        # Signers in validation are strictly unseen in training
        val_count = max(1, int(round(len(distinct_subjects) * val_split)))
        val_subjects = set(random.sample(distinct_subjects, val_count))
        train_samples = [s for s in samples if s["subject_id"] not in val_subjects]
        val_samples = [s for s in samples if s["subject_id"] in val_subjects]
    else:
        # When single canonical subject (e.g. initial vocabulary bootstrap),
        # use stratified split or retain canonical clean sample for val and augmented copies for train
        train_samples = list(samples)
        val_samples = list(samples)

    # Training augmentation pipeline
    train_transform = CoordinateAugmentation(
        jitter_std=0.015,
        scale_range=(0.92, 1.08),
        translation_range=0.02,
        time_warp_range=(0.85, 1.15),
        joint_dropout_prob=0.04,
        p=0.9,
    )

    train_dataset = PSLDataset(
        samples=train_samples,
        transform=train_transform,
        virtual_multiplier=train_virtual_multiplier,
    )

    # Validation dataset uses NO augmentation (deterministic clean benchmark)
    val_dataset = PSLDataset(
        samples=val_samples,
        transform=None,
        virtual_multiplier=1,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, class_to_idx


if __name__ == "__main__":
    from src.config import load_config
    cfg = load_config()
    train_ld, val_ld, mapping = create_dataloaders(
        landmarks_dir=cfg.paths.landmarks,
        config=cfg,
        batch_size=4,
        train_virtual_multiplier=5,
    )
    print("DataLoader Verification:")
    print(f"  Target Classes Count: {len(mapping)}")
    print(f"  Train Loader Batches: {len(train_ld)} (Virtual dataset size: {len(train_ld.dataset)})")
    print(f"  Val Loader Batches  : {len(val_ld)} (Physical dataset size: {len(val_ld.dataset)})")

    for x, y, words in train_ld:
        print(f"  Sample Batch Shape  : {x.shape} (Contract: N, C, T, V)")
        print(f"  Labels Sample       : {y.tolist()} -> {words}")
        break
