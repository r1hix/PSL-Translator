"""Neural network architecture module for Pakistani Sign Language translation.

Exports:
    - STGCN: Spatial-Temporal Graph Convolutional Network (Primary Architecture).
    - BaselineBiLSTM: Bidirectional LSTM (Academic Benchmark Baseline).
    - PSLGraph: 55-node skeletal graph topology with K=3 spatial partitioning.
    - build_model: Architecture factory utility.
"""

from typing import Any
from src.models.graph import PSLGraph


def __getattr__(name: str) -> Any:
    if name == "BaselineBiLSTM":
        from src.models.baseline_lstm import BaselineBiLSTM
        return BaselineBiLSTM
    elif name in ("STGCN", "build_model"):
        from src.models.st_gcn import STGCN, build_model
        return STGCN if name == "STGCN" else build_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "PSLGraph",
    "STGCN",
    "BaselineBiLSTM",
    "build_model",
]
