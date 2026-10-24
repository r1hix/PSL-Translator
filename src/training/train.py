"""Model Training and Optimization Pipeline for PSL-Translator.

Supports both ST-GCN and Bi-LSTM models with hardware optimization:
    - Automatic Mixed Precision (FP16 AMP) for GTX 1660 Super 6GB GPU.
    - Dynamic device resolution (CUDA on PC, MPS on Apple Silicon, CPU fallback).
    - Label-smoothed CrossEntropyLoss and CosineAnnealingLR scheduling.
    - Academic checkpointing with Top-1 Accuracy and Macro-F1 logging.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.config import load_config, Config
from src.models.baseline_lstm import BaselineBiLSTM
from src.models.st_gcn import STGCN
from src.training.dataset import create_dataloaders
from src.training.evaluate import evaluate_model, format_confusion_matrix_ascii
from src.utils import get_device, validate_environment


def train_one_epoch(
    model: nn.Module,
    train_loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    use_amp: bool = False,
) -> Tuple[float, float]:
    """Executes a single training epoch with optional FP16 mixed precision.
    
    Args:
        model: Neural network model.
        train_loader: Training DataLoader.
        optimizer: AdamW optimizer.
        criterion: CrossEntropyLoss.
        device: Target execution device.
        scaler: PyTorch GradScaler for AMP.
        use_amp: Flag to enable mixed precision.
        
    Returns:
        Tuple[float, float]: (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for x_batch, y_batch, _ in train_loader:
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp and device.type == "cuda" and scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * x_batch.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

    epoch_loss = running_loss / max(1, total)
    epoch_acc = correct / max(1, total)
    return epoch_loss, epoch_acc


def train_model(
    model_type: str = "st_gcn",
    epochs: int = 40,
    batch_size: int = 16,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    use_amp: bool = True,
    train_virtual_multiplier: int = 15,
    config: Optional[Config] = None,
) -> Dict[str, float]:
    """Trains a sign language model and saves best checkpoint.
    
    Args:
        model_type: 'st_gcn' or 'lstm'.
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
        lr: Initial learning rate.
        weight_decay: Weight decay penalty for AdamW.
        use_amp: Enable FP16 AMP on CUDA devices.
        train_virtual_multiplier: Virtual data augmentation multiplier per epoch.
        config: Loaded project config.
        
    Returns:
        Dict: Final validation metrics summary.
    """
    if config is None:
        config = load_config()

    device = get_device()
    validate_environment(workload="training")

    print(f"[*] Initializing Training Pipeline:")
    print(f"    Architecture  : {model_type.upper()}")
    print(f"    Target Device : {device.type.upper()}")
    print(f"    Batch Size    : {batch_size}")
    print(f"    Total Epochs  : {epochs}")
    print(f"    FP16 AMP      : {use_amp and device.type == 'cuda'}")

    # Build DataLoaders
    train_loader, val_loader, class_to_idx = create_dataloaders(
        landmarks_dir=config.paths.landmarks,
        config=config,
        batch_size=batch_size,
        val_split=config.training.validation_split,
        train_virtual_multiplier=train_virtual_multiplier,
        seed=config.training.seed,
    )

    class_names = [config.get_word_by_id(i).name for i in range(config.num_classes)]
    num_classes = config.num_classes

    # Instantiate Model
    if model_type.lower() == "st_gcn":
        model = STGCN(
            in_channels=3,
            num_classes=num_classes,
            graph_strategy="spatial",
            edge_importance_weighting=True,
            dropout=0.3,
        ).to(device)
    elif model_type.lower() in ("lstm", "bi_lstm"):
        model = BaselineBiLSTM(
            num_nodes=55,
            in_channels=3,
            hidden_size=128,
            num_layers=2,
            num_classes=num_classes,
            dropout=0.3,
        ).to(device)
    else:
        raise ValueError(f"Unknown architecture: {model_type}. Options: 'st_gcn', 'lstm'.")

    # Optimization Setup
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler = torch.cuda.amp.GradScaler() if (use_amp and device.type == "cuda") else None

    # Checkpoint Destination
    checkpoints_dir = config.paths.checkpoints
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = checkpoints_dir / f"{model_type}_best.pth"

    best_val_f1 = -1.0
    best_val_acc = 0.0

    print("\n" + "=" * 80)
    print(f"{'Epoch':^7} | {'Train Loss':^11} | {'Train Acc':^10} | {'Val Loss':^10} | {'Val Acc':^10} | {'Val F1':^9} | {'Time':^7}")
    print("=" * 80)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
        )

        scheduler.step()

        # Validation Pass
        val_metrics = evaluate_model(
            model=model,
            data_loader=val_loader,
            device=device,
            class_names=class_names,
            criterion=criterion,
        )

        val_loss = float(val_metrics["loss"])
        val_acc = float(val_metrics["accuracy"])
        val_f1 = float(val_metrics["macro_f1"])
        elapsed = time.time() - t0

        saved_marker = ""
        if val_f1 > best_val_f1 or (val_f1 == best_val_f1 and val_acc > best_val_acc):
            best_val_f1 = val_f1
            best_val_acc = val_acc
            saved_marker = " [*BEST]"

            # Save full state dict with metadata for viva defense
            torch.save({
                "epoch": epoch,
                "model_type": model_type,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_f1": val_f1,
                "class_names": class_names,
                "num_classes": num_classes,
            }, best_checkpoint_path)

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs or saved_marker:
            print(f"{epoch:^7d} | {train_loss:^11.4f} | {train_acc*100:^9.1f}% | {val_loss:^10.4f} | {val_acc*100:^9.1f}% | {val_f1:^9.4f} | {elapsed:^6.2f}s{saved_marker}")

    print("=" * 80)
    print(f"[+] Training Completed.")
    print(f"    Best Validation Accuracy : {best_val_acc*100:.2f}%")
    print(f"    Best Validation Macro-F1 : {best_val_f1:.4f}")
    print(f"    Checkpoint Saved To      : {best_checkpoint_path.resolve()}")

    # Load best weights for final academic evaluation
    if best_checkpoint_path.exists():
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    final_eval = evaluate_model(
        model=model,
        data_loader=val_loader,
        device=device,
        class_names=class_names,
    )

    print("\n" + "=" * 80)
    print("FINAL VALIDATION REPORT & CONFUSION MATRIX:")
    print("=" * 80)
    print(format_confusion_matrix_ascii(final_eval["confusion_matrix"], class_names))
    print(f"\nMean Inference Latency: {final_eval['mean_latency_ms']:.2f} ms ({final_eval['fps_throughput']:.1f} FPS)")

    return {
        "best_acc": best_val_acc,
        "best_f1": best_val_f1,
        "mean_latency_ms": final_eval["mean_latency_ms"],
        "checkpoint": str(best_checkpoint_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Train PSL Translation Models (ST-GCN / Bi-LSTM)")
    parser.add_argument("--model", type=str, default="st_gcn", choices=["st_gcn", "lstm"], help="Model architecture")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--virtual_multiplier", type=int, default=15, help="Virtual augmentation multiplier per epoch")
    args = parser.parse_args()

    train_model(
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        train_virtual_multiplier=args.virtual_multiplier,
    )


if __name__ == "__main__":
    main()
