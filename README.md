# Real-Time Pakistani Sign Language (PSL) Translator

An assistive deep learning system that translates fluid, dynamic Pakistani Sign Language (PSL) gestures into real-time Urdu text and synthetic speech using standard webcams.

---

## 📌 Motivation & Problem Statement

* **The Communication Barrier:** Over **1.5 million deaf individuals in Pakistan** communicate using Pakistan Sign Language (PSL). However, because PSL is a historically low-resource language, real-time assistive translation tools are virtually non-existent.
* **The Failure of Prior Work:** Most existing student and academic projects only perform classification on static alphabet pictures (fingerspelling). Real-world sign language consists of **continuous, dynamic physical motion** of hands, fingers, and upper-body gestures.
* **The Environmental Fragility Problem:** Traditional computer vision models process raw RGB video frames, making them fragile when subjected to cluttered backgrounds, poor lighting, different skin tones, or wrist accessories (e.g., watches, bracelets).

---

## 💡 The Core Innovation: Coordinate-Space Modeling

Rather than analyzing raw image pixels, this project decouples gesture recognition into a robust two-stage pipeline:

```
  [ Webcam Feed ] 
         │  (Standard 30 FPS RGB)
         ▼
  [ Stage 1: MediaPipe Holistic ] ──► Extracts 3D Anatomical Joints (X, Y, Z)
         │                           • Backgrounds, lighting & skin color discarded
         ▼
  [ Geometric Normalization ]     ──► Translation & Scale Invariance
         │                           • Centered relative to root joint
         │                           • Normalized by shoulder/palm width
         ▼
  [ Stage 2: ST-GCN Neural Net ]  ──► Spatial-Temporal Graph Convolutions
         │                           • Spatial graph captures bone connectivity
         │                           • Temporal convolutions capture gesture velocity
         ▼
  [ Bilingual Output Engine ]     ──► Urdu Text Display + Real-Time Spoken Audio
```

### Why This Approach Works Anywhere:
1. **Lighting & Background Invariant:** Background clutter, moving objects, and dim lighting are discarded at Stage 1.
2. **Skin Tone & Accessory Invariant:** MediaPipe tracks skeletal joint pivots, completely ignoring jewelry, wristwatches, or skin tone variations.
3. **Distance Invariant:** Coordinate normalization ensures the AI receives the exact same normalized motion tensor whether the user sits 2 feet or 6 feet away from the camera.

---

## 🎯 Target Vocabulary & Scope

For a focused, high-precision 12-week implementation, the system targets **15–20 essential dynamic words and phrases** derived from the official Deaf Reach Pakistan portal (`psl.org.pk`):

* **Greetings & Civility:** *Assalam-o-Alaikum* (السلام علیکم), *Shukriya* (شکریہ), *Aap kaise hain?* (آپ کیسے ہیں؟), *Theek* (ٹھیک), *Khuda Hafiz* (خدا حافظ).
* **Everyday Essentials:** *Pani* (پانی), *Khana* (کھانا), *Ghar* (گھر), *Madad* (مدد), *Haan* (ہاں), *Nahi* (نہیں۔).
* **Emergency & Health:** *Doctor* (ڈاکٹر), *Dard* (درد), *Dawa* (دوا), *Hospital* (ہسپتال).

---

## 🧠 Model Architecture: ST-GCN

The gesture sequence is modeled as a dynamic skeletal graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$:
* **Nodes ($\mathcal{V}$):** 55 skeletal joints (21 left hand + 21 right hand + 13 upper-body pose keypoints).
* **Edges ($\mathcal{E}$):** Anatomical bone connections (connecting knuckles to fingertips, wrists to elbows, etc.).
* **Spatial Convolutions:** Aggregate joint neighborhood features using the anatomical adjacency matrix $A$.
* **Temporal Convolutions:** Capture motion trajectory, speed, and rhythm across consecutive frames over a 30-frame temporal window.

---

## 📊 Datasets & Data Strategy

1. **[PSL Web Portal](https://psl.org.pk):** Official dictionary by Deaf Reach / Family Educational Services Foundation (FESF) with 5,000+ native signer video demonstrations.
2. **PkSLMNM Dataset (Mendeley Data):** Academic dataset featuring manual and non-manual Pakistani Sign Language gestures across multiple individuals.
3. **LUMS PSL OpenPose Dataset (Kaggle):** Pre-extracted skeletal coordinates for common Urdu words.
4. **Self-Recorded Validation Set:** Custom webcam recordings of the 15–20 target words to test cross-subject generalization for live classroom demonstration.

---

## 🛠️ Tech Stack & Hardware

* **Language:** Python 3.10+
* **Deep Learning:** PyTorch, Torchvision, PyTorch Geometric
* **Vision & Pose Tracking:** Google MediaPipe Holistic, OpenCV
* **Data Processing & Metrics:** NumPy, SciPy, Scikit-learn, Pandas
* **Speech Synthesis:** `gTTS` (Urdu audio) / `pyttsx3`, `pygame`
* **Hardware Setup:**
  * **Training Workstation:** AMD Ryzen 5 5600 + NVIDIA GeForce GTX 1660 Super (6GB CUDA, FP16 accelerated)
  * **Inference / Live Demo:** Apple MacBook Pro (M2 2023, Metal Performance Shaders / MPS)

---

## 📂 Project Repository Structure

```text
project/
├── README.md
├── requirements.txt
├── presentation/
│   └── Real-Time PSL Translator.pptx
├── configs/
│   └── config.yaml               # Hyperparameters, window size (30 frames), target words
├── data/
│   ├── raw_videos/               # Downloaded MP4 clips organized by class
│   ├── landmarks/                # Extracted (T, V, C) .npy coordinate sequences
│   └── audio_cache/              # Pre-generated Urdu pronunciation audio files
├── src/
│   ├── data_collection/          # Scripts to record custom clips and download PSL samples
│   ├── preprocessing/            # MediaPipe extraction and geometric normalization
│   ├── models/                   # Bi-LSTM baseline and ST-GCN PyTorch implementations
│   ├── training/                 # Training loops, loss functions, and evaluation metrics
│   └── inference/                # Real-time webcam OpenCV HUD and Urdu TTS playback
└── notebooks/
    └── 01_exploratory_analysis.ipynb
```

---

## 📅 12-Week Milestone Roadmap

* **Weeks 1–2 (Data Acquisition):** Download target vocabulary clips from `psl.org.pk`, setup custom recording script, and curate dataset.
* **Weeks 3–4 (Preprocessing & Normalization):** Build MediaPipe Holistic pipeline; implement root-centering and scale normalization; save `.npy` tensors.
* **Weeks 5–7 (Modeling & Training):** Implement baseline LSTM and proposed ST-GCN; train models on GTX 1660 Super via PyTorch CUDA.
* **Weeks 8–9 (Ablation & Optimization):** Benchmark ST-GCN vs. baselines; test noise robustness; generate confusion matrices and classification reports.
* **Weeks 10–11 (Live HUD & Audio Engine):** Build real-time webcam inference engine with OpenCV HUD overlay and low-latency Urdu speech playback.
* **Week 12 (Final Report & Viva Defense):** Perform in-person live demonstration, finalize academic report, and deliver presentation.
