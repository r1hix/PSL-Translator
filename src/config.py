"""Centralized configuration loader and validator for PSL-Translator.

Loads and parses `configs/config.yaml` into strongly-typed dataclasses and
provides project-wide absolute directory resolution without hardcoded paths.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


# Root directory of the repository (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class VocabularyItem:
    """Represents a single Pakistani Sign Language vocabulary word."""
    id: int
    name: str
    urdu: str
    roman: str
    sentence_en: str
    sentence_ur: str
    sentence_roman: str
    category: str


@dataclass(frozen=True)
class DataConfig:
    """Tensor dimensions and landmark extraction parameters."""
    fps: int
    sequence_length: int
    num_nodes: int
    num_channels: int
    left_hand_range: tuple[int, int]
    right_hand_range: tuple[int, int]
    pose_range: tuple[int, int]
    pose_indices: list[int]
    root_joint: str
    scale_reference: str
    epsilon: float


@dataclass(frozen=True)
class PathsConfig:
    """Resolved absolute paths for datasets, caches, and checkpoints."""
    raw_videos: Path
    landmarks: Path
    audio_cache: Path
    checkpoints: Path
    logs: Path


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters for model training."""
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    lr_step_milestones: list[int]
    lr_gamma: float
    use_amp: bool
    num_workers: int
    validation_split: float
    subject_split: bool
    seed: int


@dataclass(frozen=True)
class InferenceConfig:
    """Settings for real-time live webcam inference and HUD."""
    confidence_threshold: float
    consecutive_frames_trigger: int
    webcam_index: int
    frame_width: int
    frame_height: int
    display_fps: bool
    draw_landmarks: bool
    audio_enabled: bool


@dataclass(frozen=True)
class Config:
    """Top-level unified project configuration."""
    project_name: str
    version: str
    vocabulary: list[VocabularyItem]
    data: DataConfig
    paths: PathsConfig
    training: TrainingConfig
    inference: InferenceConfig
    raw_dict: dict[str, Any]

    @property
    def num_classes(self) -> int:
        """Returns total number of target vocabulary classes."""
        return len(self.vocabulary)

    def get_word_by_id(self, class_id: int) -> Optional[VocabularyItem]:
        """Look up vocabulary metadata by numeric class ID."""
        for item in self.vocabulary:
            if item.id == class_id:
                return item
        return None

    def get_word_by_name(self, name: str) -> Optional[VocabularyItem]:
        """Look up vocabulary metadata by canonical English name."""
        name_clean = name.strip().lower()
        for item in self.vocabulary:
            if item.name == name_clean:
                return item
        return None


def load_config(config_path: Optional[Path] = None) -> Config:
    """Loads configuration from YAML file.
    
    Args:
        config_path: Optional explicit Path to config.yaml. Defaults to configs/config.yaml.
        
    Returns:
        Config: Strongly-typed configuration instance.
    """
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    vocab = [
        VocabularyItem(
            id=v["id"],
            name=v["name"],
            urdu=v["urdu"],
            roman=v["roman"],
            sentence_en=v.get("sentence_en", v["roman"]),
            sentence_ur=v.get("sentence_ur", v["urdu"]),
            sentence_roman=v.get("sentence_roman", v["roman"]),
            category=v["category"],
        )
        for v in raw["vocabulary"]
    ]

    nodes_cfg = raw["data"]["nodes"]
    data_cfg = DataConfig(
        fps=raw["data"]["fps"],
        sequence_length=raw["data"]["sequence_length"],
        num_nodes=raw["data"]["num_nodes"],
        num_channels=raw["data"]["num_channels"],
        left_hand_range=(nodes_cfg["left_hand"]["start_idx"], nodes_cfg["left_hand"]["end_idx"]),
        right_hand_range=(nodes_cfg["right_hand"]["start_idx"], nodes_cfg["right_hand"]["end_idx"]),
        pose_range=(nodes_cfg["upper_body_pose"]["start_idx"], nodes_cfg["upper_body_pose"]["end_idx"]),
        pose_indices=nodes_cfg["upper_body_pose"]["pose_indices"],
        root_joint=raw["data"]["normalization"]["root_joint"],
        scale_reference=raw["data"]["normalization"]["scale_reference"],
        epsilon=float(raw["data"]["normalization"]["epsilon"]),
    )

    paths_dict = raw["paths"]
    paths_cfg = PathsConfig(
        raw_videos=PROJECT_ROOT / paths_dict["raw_videos_dir"],
        landmarks=PROJECT_ROOT / paths_dict["landmarks_dir"],
        audio_cache=PROJECT_ROOT / paths_dict["audio_cache_dir"],
        checkpoints=PROJECT_ROOT / paths_dict["checkpoints_dir"],
        logs=PROJECT_ROOT / paths_dict["logs_dir"],
    )

    # Ensure necessary runtime directories exist
    paths_cfg.raw_videos.mkdir(parents=True, exist_ok=True)
    paths_cfg.landmarks.mkdir(parents=True, exist_ok=True)
    paths_cfg.audio_cache.mkdir(parents=True, exist_ok=True)
    paths_cfg.checkpoints.mkdir(parents=True, exist_ok=True)
    paths_cfg.logs.mkdir(parents=True, exist_ok=True)

    train_dict = raw["training"]
    train_cfg = TrainingConfig(
        batch_size=train_dict["batch_size"],
        epochs=train_dict["epochs"],
        learning_rate=train_dict["learning_rate"],
        weight_decay=train_dict["weight_decay"],
        lr_step_milestones=train_dict["lr_step_milestones"],
        lr_gamma=train_dict["lr_gamma"],
        use_amp=train_dict["use_amp"],
        num_workers=train_dict["num_workers"],
        validation_split=train_dict["validation_split"],
        subject_split=train_dict["subject_split"],
        seed=train_dict["seed"],
    )

    inf_dict = raw.get("inference", {})
    inf_cfg = InferenceConfig(
        confidence_threshold=float(inf_dict.get("confidence_threshold", 0.75)),
        consecutive_frames_trigger=int(inf_dict.get("consecutive_frames_trigger", 5)),
        webcam_index=int(inf_dict.get("webcam_index", 0)),
        frame_width=int(inf_dict.get("frame_width", 1280)),
        frame_height=int(inf_dict.get("frame_height", 720)),
        display_fps=bool(inf_dict.get("display_fps", True)),
        draw_landmarks=bool(inf_dict.get("draw_landmarks", True)),
        audio_enabled=bool(inf_dict.get("audio_enabled", True)),
    )

    return Config(
        project_name=raw["project"]["name"],
        version=raw["project"]["version"],
        vocabulary=vocab,
        data=data_cfg,
        paths=paths_cfg,
        training=train_cfg,
        inference=inf_cfg,
        raw_dict=raw,
    )
