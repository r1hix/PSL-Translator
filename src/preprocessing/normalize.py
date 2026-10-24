"""Geometric coordinate normalization and temporal resampling engine.

Academic Explainability:
    Transforms raw pixel-relative MediaPipe landmark coordinates into an
    anatomically scale-invariant and translation-invariant coordinate space:
    1. Root-Centering: Translation invariance relative to mid-shoulder chest center.
    2. Shoulder-Scaling: Scale invariance normalized by Euclidean shoulder span.
    3. Temporal Resampling: Uniform temporal interpolation to fixed window T=30 frames.
    4. Missing Joint Imputation: Robust NaN prevention and zero/wrist-relative padding.
"""

import numpy as np
from typing import Optional, Tuple


# Default anatomical indices within 55-node array
LEFT_SHOULDER_IDX = 47
RIGHT_SHOULDER_IDX = 48
LEFT_HIP_IDX = 53
RIGHT_HIP_IDX = 54
NOSE_IDX = 42

LEFT_HAND_SLICE = slice(0, 21)
RIGHT_HAND_SLICE = slice(21, 42)
POSE_SLICE = slice(42, 55)


def center_coordinates(
    landmarks: np.ndarray,
    left_shoulder_idx: int = LEFT_SHOULDER_IDX,
    right_shoulder_idx: int = RIGHT_SHOULDER_IDX,
) -> Tuple[np.ndarray, np.ndarray]:
    """Centers 3D landmarks relative to the anatomical mid-shoulder chest root joint.
    
    Mathematical Formulation:
        P_chest = (P_left_shoulder + P_right_shoulder) / 2
        P_centered = P - P_chest
        
    Args:
        landmarks: Coordinate array of shape (V, 3) or (T, V, 3).
        left_shoulder_idx: Index of left shoulder node (default 47).
        right_shoulder_idx: Index of right shoulder node (default 48).
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (centered_landmarks, root_center_offset)
    """
    is_sequence = landmarks.ndim == 3
    coords = landmarks.copy()

    if is_sequence:
        # Shape: (T, V, 3)
        l_shoulder = coords[:, left_shoulder_idx, :]  # (T, 3)
        r_shoulder = coords[:, right_shoulder_idx, :]  # (T, 3)
        
        # Check shoulder visibility (non-zero coordinates)
        valid_shoulders = (np.linalg.norm(l_shoulder, axis=-1) > 1e-4) & (np.linalg.norm(r_shoulder, axis=-1) > 1e-4)
        
        root_center = np.zeros_like(l_shoulder)
        root_center[valid_shoulders] = (l_shoulder[valid_shoulders] + r_shoulder[valid_shoulders]) / 2.0
        
        # Fallback to nose for frames with missing shoulders
        missing = ~valid_shoulders
        if np.any(missing):
            nose = coords[:, NOSE_IDX, :]
            root_center[missing] = nose[missing]

        centered = coords - root_center[:, np.newaxis, :]
        return centered, root_center
    else:
        # Shape: (V, 3)
        l_shoulder = coords[left_shoulder_idx]
        r_shoulder = coords[right_shoulder_idx]

        if np.linalg.norm(l_shoulder) > 1e-4 and np.linalg.norm(r_shoulder) > 1e-4:
            root_center = (l_shoulder + r_shoulder) / 2.0
        else:
            root_center = coords[NOSE_IDX]

        centered = coords - root_center
        return centered, root_center


def scale_normalize(
    landmarks: np.ndarray,
    left_shoulder_idx: int = LEFT_SHOULDER_IDX,
    right_shoulder_idx: int = RIGHT_SHOULDER_IDX,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Normalizes landmarks by the Euclidean distance between left and right shoulders.
    
    Mathematical Formulation:
        D_shoulder = ||P_left_shoulder - P_right_shoulder||_2
        P_norm = P_centered / (D_shoulder + epsilon)
        
    Args:
        landmarks: Centered coordinate array of shape (V, 3) or (T, V, 3).
        left_shoulder_idx: Index of left shoulder node.
        right_shoulder_idx: Index of right shoulder node.
        epsilon: Small numerical stability term to avoid division by zero.
        
    Returns:
        np.ndarray: Scale-invariant coordinate array with preserved shape.
    """
    coords = landmarks.copy()
    is_sequence = coords.ndim == 3

    if is_sequence:
        # Compute distance per frame: shape (T, 1, 1)
        l_sh = coords[:, left_shoulder_idx, :]
        r_sh = coords[:, right_shoulder_idx, :]
        dist = np.linalg.norm(l_sh - r_sh, axis=-1, keepdims=True)  # (T, 1)
        
        # If shoulder distance is invalid/collapsed, fallback to median valid distance or 1.0
        valid_dist = dist[dist > 1e-3]
        fallback_scale = np.median(valid_dist) if len(valid_dist) > 0 else 1.0
        dist = np.where(dist > 1e-3, dist, fallback_scale)
        
        scale_factor = dist[:, np.newaxis, :] + epsilon  # (T, 1, 1)
        return coords / scale_factor
    else:
        l_sh = coords[left_shoulder_idx]
        r_sh = coords[right_shoulder_idx]
        dist = np.linalg.norm(l_sh - r_sh)
        if dist < 1e-3:
            dist = 1.0
        return coords / (dist + epsilon)


def impute_missing_hands(
    sequence: np.ndarray,
    left_wrist_pose_idx: int = 51,
    right_wrist_pose_idx: int = 52,
) -> np.ndarray:
    """Imputes missing hand landmarks to ensure continuous mathematical manifolds without NaNs.
    
    Academic Explainability:
        If a hand is undetected by MediaPipe in certain frames (e.g. resting at side or
        temporarily occluded), coordinates default to 0. Zeroing coordinates after centering
        produces an artificial negative jump. Here, missing hands are placed at the corresponding
        pose wrist coordinate with zero relative displacement.
        
    Args:
        sequence: Normalized sequence array of shape (T, 55, 3).
        left_wrist_pose_idx: Node index for pose left wrist (idx 51).
        right_wrist_pose_idx: Node index for pose right wrist (idx 52).
        
    Returns:
        np.ndarray: Cleaned sequence array without abrupt zero-discontinuities.
    """
    seq = sequence.copy()
    t_steps = seq.shape[0]

    for t in range(t_steps):
        # Left Hand (indices 0 to 20)
        lh = seq[t, LEFT_HAND_SLICE, :]
        if np.all(lh == 0.0) or np.linalg.norm(lh) < 1e-4:
            # Place hand cluster at the wrist position
            seq[t, LEFT_HAND_SLICE, :] = seq[t, left_wrist_pose_idx, :]

        # Right Hand (indices 21 to 41)
        rh = seq[t, RIGHT_HAND_SLICE, :]
        if np.all(rh == 0.0) or np.linalg.norm(rh) < 1e-4:
            seq[t, RIGHT_HAND_SLICE, :] = seq[t, right_wrist_pose_idx, :]

    return seq


def resample_temporal_sequence(
    sequence: np.ndarray,
    target_length: int = 30,
) -> np.ndarray:
    """Uniformly resamples a variable-length frame sequence to exact target length T.
    
    Academic Explainability:
        Uses piece-wise 1D linear interpolation along the temporal axis t in [0, 1]
        to map varying frame rates and signing speeds to fixed-dimension tensors (30, 55, 3).
        
    Args:
        sequence: Landmark sequence of shape (F, V, C) where F is source frame count.
        target_length: Desired output temporal dimension T (default: 30).
        
    Returns:
        np.ndarray: Resampled tensor of shape (target_length, V, C) in float32.
    """
    src_len, num_nodes, num_channels = sequence.shape
    if src_len == target_length:
        return sequence.astype(np.float32)

    if src_len < 2:
        # Replicate single frame to match target length
        return np.repeat(sequence, target_length, axis=0).astype(np.float32)

    src_times = np.linspace(0.0, 1.0, num=src_len, endpoint=True)
    target_times = np.linspace(0.0, 1.0, num=target_length, endpoint=True)

    resampled = np.zeros((target_length, num_nodes, num_channels), dtype=np.float32)

    # Interpolate each joint and coordinate channel independently across time
    for v in range(num_nodes):
        for c in range(num_channels):
            resampled[:, v, c] = np.interp(target_times, src_times, sequence[:, v, c])

    return resampled


def normalize_sequence(
    raw_sequence: np.ndarray,
    target_length: int = 30,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Full normalization pipeline converting raw video landmarks into model-ready tensors.
    
    Pipeline Stages:
        1. Temporal Resampling -> (30, 55, 3)
        2. Root-Centering -> Center at chest midpoint
        3. Shoulder-Scale Normalization -> Divide by shoulder distance
        4. Hand Imputation -> Anchor missing hands to pose wrists
        
    Args:
        raw_sequence: Extracted landmarks of shape (F, 55, 3).
        target_length: Target frame length T (default: 30).
        epsilon: Numerical stability constant.
        
    Returns:
        np.ndarray: Normalized tensor of shape (30, 55, 3) with dtype float32.
    """
    # 1. Resample to fixed window T = 30
    resampled = resample_temporal_sequence(raw_sequence, target_length=target_length)

    # 2. Root Centering
    centered, _ = center_coordinates(resampled)

    # 3. Shoulder Scale Normalization
    scaled = scale_normalize(centered, epsilon=epsilon)

    # 4. Impute Missing Hands
    normalized = impute_missing_hands(scaled)

    # Replace any accidental residual NaNs with zeros
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)

    return normalized.astype(np.float32)
