# SPECS.md — Technical Roadmap & Agent Checklist

## 1. Machine Matrix & Workload Allocation
Always inspect OS (`platform.system()`) and hardware before running tasks:

| Workload | Recommended Rig | Hardware Specs | Trigger to Suggest Switch |
| :--- | :--- | :--- | :--- |
| **Model Training & Extraction** | **PC Rig (Windows)** | AMD Ryzen 5 5600 + GTX 1660 Super (6GB CUDA, FP16) | Heavy training triggered on Mac |
| **Live Demo & OpenCV HUD** | **Mac Rig (macOS)** | Apple MacBook Pro M2 (Apple Silicon MPS, built-in webcam) | Webcam demo run on headless/PC rig |

> **Agent Rule:** If current host does not match recommended rig for a task, warn the user and suggest switching before execution.

---

## 2. Git Protocol
* **Branch Policy:** All changes MUST occur on `feature/<name>` or `fix/<name>`. Never commit directly to `main`.
* **Commits:** Conventional commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`).
* **Merge Rule:** **DO NOT** merge into `main` unless the user explicitly requests a merge in that very prompt.

---

## 3. Data Contract Summary
* **Landmarks ($V=55$):** Left Hand (0-20), Right Hand (21-41), Upper Pose (42-54).
* **Sequence Tensor:** $(T=30, V=55, C=3)$ float32, centered at chest, scale-normalized by shoulder width.
* **PyTorch Batch:** $(N, 3, 30, 55)$.

---

## 4. Implementation Roadmap & Checklist

### Phase 1: Project Scaffolding & Configuration
- [x] Create directory tree: `configs/`, `data/`, `src/` modules, `notebooks/`.
- [x] Implement `configs/config.yaml`: 15-20 target words, hyperparams, tensor specs, paths.
- [x] Create module `__init__.py` files with proper exports.

### Phase 2: Data Collection Tooling
- [x] Implement `src/data_collection/psl_scraper.py`: Fetch target word videos from `psl.org.pk`.
- [x] Implement `src/data_collection/record_samples.py`: Interactive 30-frame webcam recorder with countdown.
- [x] Implement validation check for collected video integrity (`verify_data.py`).


### Phase 3: Landmark Extraction & Normalization
- [x] Implement `src/preprocessing/extract_landmarks.py`: MediaPipe Holistic 55-keypoint extractor.
- [x] Implement `src/preprocessing/normalize.py`: Translation centering, shoulder scale normalization, hand imputation.
- [x] Implement batch extractor converting raw MP4s to `.npy` $(30, 55, 3)$ files.


### Phase 4: Model Modeling (ST-GCN + Bi-LSTM)
- [ ] Implement `src/models/graph.py`: 55-node skeletal adjacency matrix $A$ (intra-hand + upper body bones).
- [ ] Implement `src/models/st_gcn.py`: Spatial Graph Conv + Temporal Conv blocks $(N, 3, 30, 55)$.
- [ ] Implement `src/models/baseline_lstm.py`: Bi-LSTM baseline for academic viva comparison.

### Phase 5: Training & Validation Pipeline
- [ ] Implement `src/training/dataset.py`: PyTorch `Dataset` with subject-split cross validation.
- [ ] Implement `src/training/train.py`: FP16 AMP training loop, loss functions, learning rate scheduler.
- [ ] Implement `src/training/evaluate.py`: Top-1 Accuracy, Macro-F1, per-class Confusion Matrix logging.

### Phase 6: Real-Time HUD & Spoken Urdu Engine
- [ ] Implement `src/inference/audio_engine.py`: Pre-cache Urdu MP3s via `gTTS`, non-blocking playback via `pygame.mixer`.
- [ ] Implement `src/inference/live_demo.py`: 30-frame rolling buffer, confidence smoothing, OpenCV bilingual HUD.
- [ ] Benchmark real-time latency ensuring $\ge 30$ FPS throughput.
