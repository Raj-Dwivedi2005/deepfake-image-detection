# 🛡️ Deepfake Image Detection: Multi-Modal Spectral & Deep Learning Classifier

[![CI Status](https://github.com/Raj-Dwivedi2005/deepfake-image-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/Raj-Dwivedi2005/deepfake-image-detection/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-TensorFlow%202.x%20%7C%20Scikit--Learn-orange.svg)](https://tensorflow.org/)

An end-to-end computer vision and deep learning system built to distinguish real human face photos from AI-generated/deepfake faces (StyleGAN / Generative Neural Networks). 

This project combines **handcrafted data-mining features** (2D FFT/DCT spectral energy, Local Binary Pattern texture histograms, high-pass noise residual moments, and RGB/HSV color inconsistencies) with **deep learning architectures** (Custom CNN baseline, ImageNet-pretrained EfficientNetB0, and Late Fusion models).

---

## 📌 Executive Summary & Problem Context

As generative AI models (StyleGAN, Diffusion Models, FLUX) produce photorealistic human faces, verifying image authenticity has become critical for media forensics, identity verification, and combating misinformation.

Generative neural networks introduce persistent micro-artifacts during upsampling (e.g. transposed convolutions), resulting in:
1. **Periodic spectral grid spikes** in 2D Fourier and DCT domains.
2. **Micro-texture unnatural smoothness** detectable via Local Binary Patterns (LBP).
3. **High-pass noise residual discrepancies** compared to camera sensor PRNU noise.
4. **Color space & chromatic aberration inconsistencies** in HSV channels.

This codebase systematically benchmarks classical ML models trained on these handcrafted physical signals against modern deep convolutional networks and late-fusion architectures.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             Input Face Image                                │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                         ┌─────────────┴─────────────┐
                         │ Face Detector & Alignment │
                         │ (OpenCV Haar + Fallback)  │
                         └─────────────┬─────────────┘
                                       │
                ┌──────────────────────┴──────────────────────┐
                ▼                                             ▼
  ┌────────────────────────────┐               ┌─────────────────────────────┐
  │ Handcrafted Feature Eng.   │               │ Deep Learning Pipeline      │
  │ - 2D FFT/DCT Spectral      │               │ - Custom CNN from Scratch   │
  │ - LBP Micro-Textures       │               │ - Pretrained EfficientNetB0 │
  │ - High-pass Noise Residual │               │ - Late Fusion Architecture  │
  │ - Color / HSV Histograms   │               └──────────────┬──────────────┘
  └─────────────┬──────────────┘                              │
                ▼                                             │
  ┌────────────────────────────┐                              │
  │ StandardScaler + PCA (50D) │                              │
  │ Random Forest / SVM        │                              │
  └─────────────┬──────────────┘                              │
                └──────────────────────┬──────────────────────┘
                                       ▼
                         ┌───────────────────────────┐
                         │ Evaluation & Grad-CAM     │
                         │ CLI & Streamlit Web App   │
                         └───────────────────────────┘
```

---

## 📊 Benchmark Model Performance

Metrics evaluated on held-out 15% stratified test sets with fixed random seed `42`:

| Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Description |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Classical Random Forest** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | Handcrafted FFT + LBP + Noise + Color (PCA 50D) |
| **Custom CNN (From Scratch)** | **0.9556** | **0.9565** | **0.9565** | **0.9565** | **0.9880** | 3-Block Conv2D + BatchNorm + Dropout Baseline |
| **EfficientNetB0 Transfer** | **0.9667** | **0.9778** | **0.9565** | **0.9670** | **0.9925** | ImageNet Pretrained & Fine-tuned |
| **Late Fusion Model** | **0.9778** | **0.9783** | **0.9783** | **0.9783** | **0.9950** | Concatenated Bottleneck CNN + Spectral Vector |

> All figures, ROC curves, confusion matrices, and metrics are auto-exported to `reports/metrics.json` and `reports/figures/`.

---

## 🔍 Model Explainability & Grad-CAM Heatmaps

To ensure classification decisions rely on authentic generative artifacts rather than background noise, Gradient-Weighted Class Activation Mapping (Grad-CAM) visualizes the spatial attention of CNN layers:

- **Generative Artifact Focus:** Highlights boundary unnatural textures, eye/iris symmetry anomalies, and cheek/skin grid patterns.

Sample visualization generated automatically:
`reports/figures/gradcam_sample.png`

---

## 🛠️ Repository Structure

```
deepfake-image-detection/
├── .github/workflows/ci.yml    # Continuous Integration pipeline
├── src/
│   ├── data/                   # Face detection, alignment, augmentation & splitters
│   ├── features/               # 2D FFT/DCT, LBP, Noise Residuals, Color & PCA pipeline
│   ├── models/                 # Random Forest, Custom CNN, EfficientNetB0 & Fusion
│   ├── evaluation/             # Metrics, ROC curves, Grad-CAM & visualization engine
│   ├── inference/              # CLI prediction script & Streamlit interactive web app
│   └── train_and_evaluate.py   # Master end-to-end execution script
├── tests/                      # Pytest unit test suite
├── configs/config.yaml         # Centralized hyperparameter & pipeline config
├── reports/                    # Auto-generated metrics.json and PNG figures
├── requirements.txt            # Python dependencies
├── .gitignore
├── LICENSE                     # MIT License
└── README.md                   # Project documentation
```

---

## 🚀 Quickstart & Reproduction Guide

### 1. Clone & Setup Environment

```bash
git clone https://github.com/Raj-Dwivedi2005/deepfake-image-detection.git
cd deepfake-image-detection

python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Run Unit Test Suite

```bash
pytest tests/ -v
```

### 3. Run End-to-End Training & Evaluation Pipeline

```bash
python -m src.train_and_evaluate
```

### 4. Single Image CLI Inference

```bash
python -m src.inference.predict --image path/to/face.jpg --model classical
```

### 5. Launch Interactive Streamlit Web App

```bash
streamlit run src/inference/app.py
```

---

## ⚠️ Limitations & Future Engineering Work

1. **Generalization Across Unseen Architectures:** While models achieve high accuracy on StyleGAN artifacts, performance may decay on emerging Diffusion Models (e.g. Midjourney v6, DALL-E 3) without continuous domain adaptation fine-tuning.
2. **Video Deepfake Detection:** The current pipeline focuses on static keyframes. Expanding to temporal consistency analysis (3D CNNs / Vision Transformers) across video frames is an essential future enhancement.
3. **Adversarial Robustness:** Social media platforms often re-compress or apply heavy spatial filtering. Continuing research into adversarial noise training is recommended.

---

## 📜 License

This repository is licensed under the [MIT License](LICENSE).
