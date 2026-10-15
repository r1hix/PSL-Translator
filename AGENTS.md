# AGENTS.md — Developer & AI Agent Guidelines

> **Project:** Real-Time Pakistani Sign Language (PSL) Translator  
> **Course:** CS4523 Artificial Intelligence Lab  
> **Target Framework:** PyTorch, MediaPipe Holistic, OpenCV, Spatial-Temporal Graph Convolutional Networks (ST-GCN)

---

## 🎯 1. Mission & Non-Negotiable Core Principles

All AI agents, coding assistants, and human contributors working on this repository must adhere strictly to these architectural decisions:

1. **Coordinate-Space Only (Never Raw Pixels):**
   * **Rule:** Do NOT implement raw RGB video classification (no 3D-CNNs, I3D, or raw frame ResNets).
   * **Reason:** Raw pixel models are brittle to cluttered backgrounds, wrist accessories, lighting shifts, and skin tone biases. 
   * **Pipeline:** Raw Video $\rightarrow$ MediaPipe Holistic Landmark Extraction $\rightarrow$ Normalization $\rightarrow$ Graph Modeling (ST-GCN).
2. **Academic Explainability is Mandatory:**
   * The author must present and defend every line of code in an academic viva.
   * **Rule:** Avoid obscure one-liners, undocumented hyperparameters, or "magic numbers." Every PyTorch layer, graph adjacency matrix formulation, and tensor manipulation must include concise mathematical comments explaining *why* it exists.
3. **Hardware Agnostic (Dual-Machine Compatibility):**
   * Code must seamlessly execute on:
     * **Training Rig:** PC with AMD Ryzen 5 5600 + NVIDIA GTX 1660 Super (6GB VRAM, CUDA, FP16 support).
     * **Inference/Demo Rig:** Apple MacBook Pro (M2 2023, Apple Silicon MPS acceleration).
   * Never hardcode `.cuda()` or `.to("mps")`. Always use dynamic device resolution:
     ```python
     device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
     ```

---

## 📐 2. Data Contract & Tensor Specifications

To ensure seamless integration between preprocessing, training, and real-time inference, all modules must conform to this exact tensor contract:

### Keypoint Anatomy (55 Total Joints / Nodes)
| Landmark Group | Source | Keypoint Count | Index Range |
| :--- | :--- | :--- | :--- |
| **Left Hand** | MediaPipe Holistic Hand | 21 keypoints $(x, y, z)$ | `0` to `20` |
| **Right Hand** | MediaPipe Holistic Hand | 21 keypoints $(x, y, z)$ | `21` to `41` |
| **Upper Body Pose** | MediaPipe Holistic Pose | 13 keypoints $(x, y, z)$ | `42` to `54` (Nose, eyes, shoulders, elbows, wrists) |

*Total Node Count ($V$):* **55 nodes**.

### Tensor Dimensions
* **Raw Frame:** `(H, W, 3)` (RGB image at 30 FPS).
* **Single Frame Landmark Slice:** `(55, 3)` (float32 coordinates $[x, y, z]$).
* **Extracted Sequence Window:** `(T, V, C) = (30, 55, 3)` where $T = 30$ frames (1.0 second window at 30 FPS).
* **ST-GCN PyTorch Batch Tensor:** `(N, C, T, V)` where:
  * $N$ = Batch size (default: 32 or 64)
  * $C$ = 3 (Channel dimension: $x, y, z$)
  * $T$ = 30 (Temporal sequence length)
  * $V$ = 55 (Skeletal vertices / joints)

---

## ⚙️ 3. Normalization & Preprocessing Rules

When processing keypoint arrays, the following transformations are mandatory before passing tensors to any neural model:

1. **Missing Hand Imputation:**
   * If a frame does not detect a hand, **never** insert NaNs or random noise.
   * If hand is out of frame, pad with zeros or hold the last known coordinate with a valid-joint binary mask.
2. **Translation Invariance (Root Centering):**
   * Center all coordinates relative to the mid-shoulder / chest root joint:
     $$\mathbf{P}_{\text{centered}} = \mathbf{P} - \mathbf{P}_{\text{chest}}$$
3. **Scale Invariance:**
   * Normalize all distances by the anatomical Euclidean distance between left and right shoulders ($D_{\text{shoulder}}$):
     $$\mathbf{P}_{\text{norm}} = \frac{\mathbf{P}_{\text{centered}}}{D_{\text{shoulder}} + \epsilon}$$
   * If shoulders are not clearly visible, fallback to wrist-to-middle-MCP knuckle distance.

---

## 🏗️ 4. Codebase Organization & Module Boundaries

Agents must respect module boundaries. Do not place business logic inside scripts meant for data collection or inference:

```text
src/
├── data_collection/    # Web scrapers for psl.org.pk and custom recording scripts
├── preprocessing/      # MediaPipe feature extraction and coordinate normalization
├── models/             # PyTorch model definitions ONLY (ST-GCN, Bi-LSTM baselines)
├── training/           # PyTorch Datasets, DataLoaders, train/val loops, loss definitions
└── inference/          # Real-time webcam OpenCV loop, HUD rendering, and Urdu audio synthesis
```

### Path & Configuration Rules:
* **No Hardcoded Absolute Paths:** Always use `pathlib.Path` relative to the project root.
* **Centralized Configuration:** Hyperparameters (learning rate, window size, batch size, target classes) must be defined in `configs/config.yaml`, not scattered in code.
* **Large Files:** Never store large video files (`.mp4`) or generated landmark directories (`.npy`) inside Git. Keep `data/raw_videos/` and `data/landmarks/` ignored in `.gitignore`.

---

## 💻 5. Coding Standards & Conventions

1. **Type Hints & Docstrings:** Every function and class method must include standard Python type annotations and a Google-style or NumPy-style docstring:
   ```python
   def extract_landmarks(frame: np.ndarray, holistic_model) -> np.ndarray:
       """Extracts and formats 55 skeletal landmarks from an RGB video frame.
       
       Args:
           frame: Input image array of shape (H, W, 3) in BGR/RGB format.
           holistic_model: Initialized MediaPipe Holistic solution instance.
           
       Returns:
           np.ndarray: Landmark matrix of shape (55, 3) containing normalized (x, y, z).
       """
   ```
2. **Memory Safety for 6GB GPU (GTX 1660 Super):**
   * When writing training loops, always use `torch.cuda.amp.autocast()` for FP16 mixed precision.
   * Explicitly delete intermediate tensors and call `torch.cuda.empty_cache()` if memory thresholds are approached.
   * Keep default batch size at 32 or 64.
3. **Real-Time HUD Performance:**
   * The live demo script (`src/inference/live_demo.py`) must maintain $\ge 30$ FPS.
   * Avoid calling heavy blocking I/O (like generating TTS audio on the fly) inside the OpenCV frame loop. Pre-render Urdu pronunciation MP3s into `data/audio_cache/` and trigger playback asynchronously using `pygame.mixer`.

---

## 🧪 6. Verification & Evaluation Protocol

Before submitting changes or marking tasks as complete, agents must verify:

1. **Tensor Shape Integrity:** Ensure extracted landmark matrices match `(30, 55, 3)` and PyTorch batches match `(N, 3, 30, 55)`.
2. **Baseline Comparison:** Any new graph model must be benchmarked against the standard `Bi-LSTM` baseline on the same validation split.
3. **Reporting Metrics:** Always log:
   * **Top-1 Accuracy**
   * **Macro F1-Score** (to ensure low-frequency signs aren't ignored)
   * **Per-class Confusion Matrix**
   * **Mean Inference Latency (ms)**

---

## 🚨 7. Common Pitfalls to Avoid

* ❌ **Do NOT use OpenPose** when MediaPipe Holistic is specified. MediaPipe is significantly lighter, faster on CPU/webcam, and does not require complex C++ compilation.
* ❌ **Do NOT randomize cross-validation without subject splitting** if multiple recordings come from the same person. Ensure validation splits evaluate generalization to unseen signers.
* ❌ **Do NOT over-complicate the vocabulary.** Stick strictly to the selected 15–20 high-frequency words defined in `configs/config.yaml`.
