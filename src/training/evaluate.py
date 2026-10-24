"""Evaluation and metric computation for PSL translation models.

Computes academic performance metrics compliant with the evaluation protocol:
    1. Top-1 Accuracy
    2. Macro-averaged F1-Score
    3. Per-class Precision, Recall, and F1
    4. Multi-class Confusion Matrix
    5. Mean Inference Latency (ms/sample) to ensure >= 30 FPS compliance
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    class_names: Optional[List[str]] = None,
    criterion: Optional[nn.Module] = None,
) -> Dict[str, Union[float, np.ndarray, str, dict]]:
    """Evaluates a PyTorch model on a given DataLoader.
    
    Args:
        model: Trained ST-GCN or Bi-LSTM model instance.
        data_loader: PyTorch DataLoader yielding (tensors, labels, words).
        device: Target execution device (cuda, mps, cpu).
        class_names: Optional list of class names ordered by class ID.
        criterion: Optional loss function for calculating validation loss.
        
    Returns:
        Dict containing accuracy, macro_f1, mean_latency_ms, confusion_matrix, and report.
    """
    model.eval()
    all_preds: List[int] = []
    all_targets: List[int] = []
    total_loss: float = 0.0
    total_batches: int = 0
    inference_latencies: List[float] = []

    for x_batch, y_batch, _ in data_loader:
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        # Measure forward inference latency
        t0 = time.perf_counter()
        logits = model(x_batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        batch_latency_ms = (t1 - t0) * 1000.0 / x_batch.size(0)
        inference_latencies.append(batch_latency_ms)

        if criterion is not None:
            loss = criterion(logits, y_batch)
            total_loss += loss.item()
            total_batches += 1

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(y_batch.cpu().tolist())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    # Top-1 Accuracy: Correct / Total
    accuracy = float(np.mean(y_true == y_pred)) if len(y_true) > 0 else 0.0

    # Macro F1-Score: harmonic mean giving equal weight to every class
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # Confusion matrix
    num_classes = len(class_names) if class_names else max(max(y_true, default=0), max(y_pred, default=0)) + 1
    labels_range = list(range(num_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels_range)

    # Detailed classification report
    report = classification_report(
        y_true,
        y_pred,
        labels=labels_range,
        target_names=class_names if class_names else [f"class_{i}" for i in labels_range],
        zero_division=0,
        output_dict=True,
    )

    avg_latency = float(np.mean(inference_latencies)) if inference_latencies else 0.0
    avg_loss = total_loss / max(1, total_batches) if criterion is not None else 0.0

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "loss": avg_loss,
        "mean_latency_ms": avg_latency,
        "fps_throughput": 1000.0 / avg_latency if avg_latency > 0 else 0.0,
        "confusion_matrix": cm,
        "classification_report": report,
    }


def format_confusion_matrix_ascii(cm: np.ndarray, class_names: List[str]) -> str:
    """Formats confusion matrix into a clean ASCII table for CLI logs and viva defense.
    
    Args:
        cm: Confusion matrix array of shape (K, K).
        class_names: List of class names.
        
    Returns:
        str: Formatted printable ASCII matrix.
    """
    lines: List[str] = []
    max_len = max((len(name[:10]) for name in class_names), default=6)
    col_w = max(4, max_len)

    # Header
    header = " " * (col_w + 2) + " ".join(f"{name[:col_w]:>{col_w}}" for name in class_names)
    lines.append(header)
    lines.append("-" * len(header))

    # Rows
    for i, name in enumerate(class_names):
        row_str = " ".join(f"{cm[i, j]:>{col_w}d}" for j in range(len(class_names)))
        lines.append(f"{name[:col_w]:>{col_w}} | {row_str}")

    return "\n".join(lines)


if __name__ == "__main__":
    from src.config import load_config
    from src.models.st_gcn import STGCN
    from src.training.dataset import create_dataloaders
    from src.utils import get_device

    cfg = load_config()
    device = get_device()
    _, val_ld, mapping = create_dataloaders(cfg.paths.landmarks, cfg, batch_size=4)
    model = STGCN(in_channels=3, num_classes=cfg.num_classes).to(device)

    names = [cfg.get_word_by_id(i).name for i in range(cfg.num_classes)]
    results = evaluate_model(model, val_ld, device=device, class_names=names)
    print("Academic Evaluation Smoke Test:")
    print(f"  Top-1 Accuracy     : {results['accuracy'] * 100:.2f}%")
    print(f"  Macro F1-Score     : {results['macro_f1']:.4f}")
    print(f"  Mean Latency       : {results['mean_latency_ms']:.2f} ms ({results['fps_throughput']:.1f} FPS)")
