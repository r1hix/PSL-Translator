"""Model training and evaluation module for PSL-Translator.

Exports:
    - PSLDataset: PyTorch dataset loading (30, 55, 3) arrays to (C, T, V) tensors.
    - CoordinateAugmentation: Domain-specific coordinate jitter, scale, and time warp.
    - create_dataloaders: Stratified / subject-independent dataloader generator.
    - evaluate_model: Top-1 Accuracy, Macro-F1, confusion matrix, and latency evaluator.
    - train_model: FP16 AMP training pipeline with checkpointing.
"""

from typing import Any
from src.training.dataset import CoordinateAugmentation, PSLDataset, create_dataloaders
from src.training.evaluate import evaluate_model, format_confusion_matrix_ascii


def __getattr__(name: str) -> Any:
    if name == "train_model":
        from src.training.train import train_model
        return train_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CoordinateAugmentation",
    "PSLDataset",
    "create_dataloaders",
    "evaluate_model",
    "format_confusion_matrix_ascii",
    "train_model",
]
