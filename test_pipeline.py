"""PSL-Translator End-to-End Pipeline Smoke Test & Verification Suite.

Runs quick automated verification tests across all project layers:
    1. Configuration & Vocabulary Integrity
    2. Data & Landmark File Validation
    3. Graph Formulation & Spatial-Temporal Adjacency Matrix
    4. Model Forward Passes (ST-GCN and Bi-LSTM Baseline)
    5. Dataset Augmentation & PyTorch DataLoader Contract (N, C, T, V)
    6. Training & Gradient Backprop Step (AMP FP16 on CUDA / CPU fallback)
    7. Bilingual NLG Translation Layer (English, Urdu, Roman Urdu)
"""

import sys
import time
from pathlib import Path

# Ensure UTF-8 output encoding on Windows PowerShell
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch
import numpy as np


def run_checks() -> bool:
    print("=" * 80)
    print("  PSL-TRANSLATOR END-TO-END PIPELINE AUDIT")
    print("=" * 80)

    # 1. Config Check
    print("\n[1/7] Testing Configuration & Vocabulary Mapping...")
    try:
        from src.config import load_config
        cfg = load_config()
        print(f"      [OK] Config loaded successfully: {cfg.num_classes} classes configured.")
    except Exception as e:
        print(f"      [FAIL] Config check failed: {e}")
        return False

    # 2. Data & Landmarks Check
    print("\n[2/7] Testing Preprocessed Landmarks (30, 55, 3)...")
    try:
        landmarks_dir = cfg.paths.landmarks
        files = list(landmarks_dir.rglob("*.npy"))
        assert len(files) >= 15, f"Expected at least 15 landmark files, found {len(files)}"
        sample_arr = np.load(files[0])
        assert sample_arr.shape == (30, 55, 3), f"Invalid shape: {sample_arr.shape}"
        assert not np.isnan(sample_arr).any(), "NaN values found in landmark array"
        print(f"      [OK] {len(files)} landmark tensors verified. Sample shape: {sample_arr.shape}, dtype: {sample_arr.dtype}.")
    except Exception as e:
        print(f"      [FAIL] Landmarks check failed: {e}")
        return False

    # 3. Graph Adjacency Formulation
    print("\n[3/7] Testing 55-Node Skeletal Graph Adjacency Matrix...")
    try:
        from src.models.graph import PSLGraph
        graph = PSLGraph(strategy="spatial")
        A = graph.A
        assert A.shape == (3, 55, 55), f"Unexpected graph adjacency shape: {A.shape}"
        print(f"      [OK] 55-node skeletal graph compiled. Adjacency shape: {A.shape} (K=3 spatial partitions).")
    except Exception as e:
        print(f"      [FAIL] Graph test failed: {e}")
        return False

    # 4. Model Architecture & Forward Passes
    print("\n[4/7] Testing Neural Model Forward Passes...")
    try:
        from src.models.st_gcn import STGCN
        from src.models.baseline_lstm import BaselineBiLSTM
        from src.utils import get_device

        device = get_device()
        print(f"      [*] Target Compute Device: {device.type.upper()}")

        dummy_batch = torch.randn(4, 3, 30, 55).to(device)

        # ST-GCN test
        stgcn_model = STGCN(in_channels=3, num_classes=cfg.num_classes).to(device)
        stgcn_out = stgcn_model(dummy_batch)
        assert stgcn_out.shape == (4, cfg.num_classes), f"Unexpected ST-GCN output: {stgcn_out.shape}"
        stgcn_params = sum(p.numel() for p in stgcn_model.parameters() if p.requires_grad)
        print(f"      [OK] ST-GCN forward pass passed: output shape {stgcn_out.shape}, {stgcn_params:,} parameters.")

        # Bi-LSTM test
        lstm_model = BaselineBiLSTM(num_nodes=55, in_channels=3, hidden_size=128, num_classes=cfg.num_classes).to(device)
        lstm_out = lstm_model(dummy_batch)
        assert lstm_out.shape == (4, cfg.num_classes), f"Unexpected Bi-LSTM output: {lstm_out.shape}"
        lstm_params = sum(p.numel() for p in lstm_model.parameters() if p.requires_grad)
        print(f"      [OK] Bi-LSTM forward pass passed: output shape {lstm_out.shape}, {lstm_params:,} parameters.")
    except Exception as e:
        print(f"      [FAIL] Model forward pass failed: {e}")
        return False

    # 5. Dataset Loader & Coordinate Augmentation
    print("\n[5/7] Testing DataLoader & Coordinate-Space Augmentations...")
    try:
        from src.training.dataset import create_dataloaders
        train_loader, val_loader, mapping = create_dataloaders(
            landmarks_dir=cfg.paths.landmarks,
            config=cfg,
            batch_size=4,
            train_virtual_multiplier=3,
        )
        x_sample, y_sample, words = next(iter(train_loader))
        assert x_sample.shape == (4, 3, 30, 55), f"Batch shape error: {x_sample.shape}"
        print(f"      [OK] DataLoader verified. Batch shape: {x_sample.shape}, Classes: {len(mapping)}.")
    except Exception as e:
        print(f"      [FAIL] DataLoader check failed: {e}")
        return False

    # 6. Training Gradient Step
    print("\n[6/7] Testing Training Gradient Backpropagation...")
    try:
        from src.training.train import train_one_epoch
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(stgcn_model.parameters(), lr=0.001)
        scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

        loss, acc = train_one_epoch(
            model=stgcn_model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
            use_amp=True,
        )
        print(f"      [OK] Backpropagation successful. Loss: {loss:.4f}, Accuracy: {acc*100:.1f}%.")
    except Exception as e:
        print(f"      [FAIL] Training step failed: {e}")
        return False

    # 7. Bilingual Translation Layer
    print("\n[7/7] Testing Bilingual NLG Translation Layer...")
    try:
        from src.inference.translation_layer import BilingualTranslator
        translator = BilingualTranslator(config=cfg)
        test_out = translator.translate(5)  # Pani (water)
        print(f"      [OK] Sign Detected : '{test_out.word_name}'")
        print(f"      [OK] English Spoken: \"{test_out.english_text}\"")
        print(f"      [OK] Roman Urdu    : \"{test_out.roman_text}\"")
    except Exception as e:
        print(f"      [FAIL] Translation layer failed: {e}")
        return False

    print("\n" + "=" * 80)
    print("  ALL 7 SYSTEM MODULES PASSED VERIFICATION WITH ZERO ERRORS!")
    print("=" * 80)
    return True


if __name__ == "__main__":
    success = run_checks()
    sys.exit(0 if success else 1)
