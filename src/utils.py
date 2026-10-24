"""Hardware abstraction and multi-machine environment detection.

Provides dynamic hardware accelerator resolution (CUDA for GTX 1660 Super,
MPS for Apple Silicon M2, CPU fallback) and machine suitability validation.
"""

import platform
import sys
from typing import Dict, Tuple

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False


def get_device():
    """Dynamically resolves the optimal hardware device for PyTorch.
    
    Academic Explainability:
        - CUDA: NVIDIA GTX 1660 Super training workstation (FP16 AMP enabled).
        - MPS: Apple Silicon M2 Metal Performance Shaders for Mac inference.
        - CPU: Reliable mathematical fallback for standard evaluation.
        
    Returns:
        torch.device or str: Resolved hardware target.
    """
    if not TORCH_AVAILABLE:
        return "cpu (torch not installed)"
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_system_info() -> Dict[str, str]:
    """Inspects the local runtime OS and computational hardware.
    
    Returns:
        dict: System metadata including OS, architecture, GPU type, and PyTorch version.
    """
    os_name = platform.system()
    arch = platform.machine()
    python_ver = sys.version.split()[0]
    torch_ver = torch.__version__ if TORCH_AVAILABLE else "Not installed"

    device = get_device()
    gpu_name = "None (CPU)"
    device_type = getattr(device, "type", "cpu") if TORCH_AVAILABLE else "cpu"
    if device_type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
    elif device_type == "mps":
        gpu_name = "Apple Silicon M2 (Metal Performance Shaders)"

    # Categorize rig
    if os_name == "Windows" and device_type == "cuda":
        rig_type = "Training Rig (Windows PC + NVIDIA GTX 1660 Super)"
    elif os_name == "Darwin" or "mac" in platform.platform().lower():
        rig_type = "Inference Rig (Apple MacBook Pro M2)"
    else:
        rig_type = f"Generic Host ({os_name} {arch})"


    return {
        "os": os_name,
        "architecture": arch,
        "python_version": python_ver,
        "torch_version": torch_ver,
        "device": str(device),
        "gpu_name": gpu_name,
        "rig_type": rig_type,
    }


def validate_workload_rig(workload: str) -> Tuple[bool, str]:
    """Validates whether the current host matches the recommended rig for a given workload.
    
    Args:
        workload: 'training' or 'inference' or 'data_collection'
        
    Returns:
        Tuple[bool, str]: (is_optimal, advisory_message)
    """
    info = get_system_info()
    device = info["device"]
    os_name = info["os"]

    if workload == "training":
        if device == "cuda":
            return True, f"Optimal training rig detected: {info['gpu_name']} with FP16 support."
        elif device == "mps":
            return False, (
                "⚠️ Workload Advisory: Heavy model training initiated on Apple MacBook (MPS).\n"
                "For faster FP16 mixed precision training, consider switching to the Windows PC rig "
                "(AMD Ryzen 5 5600 + NVIDIA GTX 1660 Super 6GB CUDA)."
            )
        else:
            return False, (
                "⚠️ Workload Advisory: Model training running on CPU.\n"
                "Please switch to the Windows PC rig (GTX 1660 Super CUDA) for accelerated training."
            )

    elif workload == "inference":
        if device == "mps":
            return True, "Optimal live inference rig detected: Apple MacBook Pro M2."
        elif device == "cuda":
            return True, f"CUDA acceleration available for inference: {info['gpu_name']}."
        else:
            return True, "Inference running on CPU. Real-time 30 FPS HUD supported via MediaPipe."

    return True, "Host environment verified."
