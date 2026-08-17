"""Interactive Streamlit Web Application for Deepfake Image Detection.

Provides image drag-and-drop upload, face alignment preview, confidence score gauge,
handcrafted spectral/texture feature inspection, and Grad-CAM visualization.
"""

import sys
from pathlib import Path
import streamlit as st
import numpy as np
import cv2
from PIL import Image

# Ensure project root is in Python path for Streamlit Cloud imports
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

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
        "classical": "Handcrafted Features + Random Forest (Recommended)",
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
    # Save temporary uploaded file
    temp_dir = project_root / "scratch"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / uploaded_file.name
    
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Uploaded Image & Detected Face")
        img_pil = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(img_pil)
        st.image(img_pil, caption="Original Image", use_container_width=True)

        try:
            face_detector = FaceDetector()
            cropped_face = face_detector.detect_and_crop(img_np)
            st.image(cropped_face, caption="Cropped & Aligned Face (256x256)", width=256)
        except Exception as e:
            st.warning(f"Face detector warning: {e}. Using raw image resize fallback.")
            cropped_face = cv2.resize(img_np, (256, 256))
            st.image(cropped_face, caption="Resized Image (256x256)", width=256)

    with col2:
        st.subheader("📊 Analysis Results")
        with st.spinner("Analyzing spectral artifacts and neural embeddings..."):
            try:
                result = predict_image(str(temp_path), model_type=model_choice)
                
                pred_label = result["prediction"]
                confidence = result["confidence_percentage"]

                if pred_label == "Fake":
                    st.error(f"⚠️ **Prediction: AI-Generated / Fake** (Confidence: {confidence}%)")
                else:
                    st.success(f"✅ **Prediction: Real Photo** (Confidence: {confidence}%)")

                st.progress(confidence / 100.0)
                st.json(result)

            except FileNotFoundError as fnf_err:
                st.warning(f"⚠️ **Model Artifact Unavailable:** {fnf_err}")
                st.info("💡 *Tip: The Classical Random Forest model artifacts are committed and active by default.*")
            except Exception as err:
                st.error(f"❌ **Error processing image:** {err}")

        st.markdown("### 🔬 Spectral Analysis (2D FFT Magnitude)")
        try:
            gray = cv2.cvtColor(cropped_face, cv2.COLOR_RGB2GRAY)
            fft = np.fft.fftshift(np.fft.fft2(gray))
            mag = np.log(np.abs(fft) + 1e-8)
            norm_mag = (mag - np.min(mag)) / (np.max(mag) - np.min(mag) + 1e-8)
            st.image(norm_mag, caption="Normalized 2D FFT Power Spectrum (Grid Spikes indicate GAN Upsampling)", width=256)
        except Exception as fft_err:
            st.warning(f"FFT Visualization error: {fft_err}")
else:
    st.info("👆 Please upload a facial photo to begin real-time deepfake verification.")
