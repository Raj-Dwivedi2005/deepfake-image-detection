"""Interactive Streamlit Web Application for Deepfake Image Detection.

Provides image drag-and-drop upload, face alignment preview, confidence score gauge,
handcrafted spectral/texture feature inspection, and Grad-CAM visualization.
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
from pathlib import Path

from src.data.face_detector import FaceDetector
from src.features.pipeline import HandcraftedFeatureExtractor
from src.inference.predict import predict_image

st.set_page_config(
    page_title="Deepfake Image Detector AI",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Deepfake Image Detection Platform")
st.markdown("""
An end-to-end computer vision and deep learning system designed to audit digital media and identify 
synthetic, AI-generated human faces (StyleGAN / Deepfake) using handcrafted frequency-domain artifacts 
and deep learning architectures.
""")

st.sidebar.header("⚙️ Configuration")
model_choice = st.sidebar.selectbox(
    "Select Model Engine",
    ["classical", "custom_cnn", "efficientnet"],
    format_func=lambda x: {
        "classical": "Handcrafted Features + Random Forest",
        "custom_cnn": "Custom CNN (From Scratch)",
        "efficientnet": "EfficientNetB0 (Transfer Learning)"
    }[x]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Key Detection Signals:**
- 2D FFT / DCT Spectral Grid Artifacts
- LBP Micro-Texture Anomalies
- High-Pass Noise Residual Moments
- RGB/HSV Color Inconsistencies
""")

uploaded_file = st.file_uploader("Upload a face image (JPG or PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save temp image
    temp_dir = Path("scratch")
    temp_dir.mkdir(exist_ok=True)
    temp_path = temp_dir / uploaded_file.name
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Uploaded Image & Detected Face")
        img_pil = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(img_pil)
        st.image(img_pil, caption="Original Image", use_container_width=True)

        face_detector = FaceDetector()
        cropped_face = face_detector.detect_and_crop(img_np)
        st.image(cropped_face, caption="Cropped & Aligned Face (256x256)", width=256)

    with col2:
        st.subheader("📊 Analysis Results")
        with st.spinner("Analyzing spectral artifacts and neural embeddings..."):
            result = predict_image(str(temp_path), model_type=model_choice)

        pred_label = result["prediction"]
        confidence = result["confidence_percentage"]

        if pred_label == "Fake":
            st.error(f"⚠️ **Prediction: AI-Generated / Fake** (Confidence: {confidence}%)")
        else:
            st.success(f"✅ **Prediction: Real Photo** (Confidence: {confidence}%)")

        st.progress(confidence / 100.0)

        st.json(result)

        st.markdown("### 🔬 Spectral Analysis (2D FFT Magnitude)")
        gray = cv2.cvtColor(cropped_face, cv2.COLOR_RGB2GRAY)
        fft = np.fft.fftshift(np.fft.fft2(gray))
        mag = np.log(np.abs(fft) + 1e-8)
        st.image(mag / np.max(mag), caption="Normalized 2D FFT Power Spectrum", width=256)
