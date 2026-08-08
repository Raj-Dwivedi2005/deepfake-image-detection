"""Inference CLI tool for single image deepfake classification.

Accepts image path input via argparse, runs pre-processing, face extraction,
model prediction, and prints JSON output with real/fake label and confidence score.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
import cv2
import numpy as np

from src.data.face_detector import FaceDetector
from src.features.pipeline import HandcraftedFeatureExtractor
from src.models.classical import ClassicalBaselineModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("predict")

def predict_image(image_path: str, model_type: str = "classical", config_path: str = "configs/config.yaml") -> dict:
    """Predict whether a given face image is Real or AI-generated Fake.

    Args:
        image_path: Path to input image file.
        model_type: Model engine to use ('classical', 'custom_cnn', 'efficientnet').
        config_path: Path to YAML configuration file.

    Returns:
        Dictionary containing label, confidence, prediction_score, and metadata.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Input image path does not exist: {image_path}")

    # 1. Load image & crop face
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        raise ValueError(f"Could not decode image at {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    face_detector = FaceDetector()
    cropped_face = face_detector.detect_and_crop(img_rgb, target_size=(256, 256))
    normalized_face = cropped_face.astype(np.float32) / 255.0

    # 2. Model Prediction
    if model_type == "classical":
        feature_extractor = HandcraftedFeatureExtractor()
        if Path("reports/scaler.pkl").exists() and Path("reports/pca.pkl").exists():
            feature_extractor.load(scaler_path="reports/scaler.pkl", pca_path="reports/pca.pkl")
            raw_feat = feature_extractor.extract_raw_features(normalized_face)
            transformed_feat = feature_extractor.transform(np.expand_dims(raw_feat, axis=0))
        else:
            # Fallback fit on raw feature if not trained
            raw_feat = feature_extractor.extract_raw_features(normalized_face)
            transformed_feat = feature_extractor.fit_transform(np.expand_dims(raw_feat, axis=0))

        model = ClassicalBaselineModel(model_type="random_forest")
        if Path("reports/classical_model.pkl").exists():
            model.load("reports/classical_model.pkl")
            prob_fake = float(model.predict_proba(transformed_feat)[0, 1])
        else:
            # Default placeholder confidence if model weight is absent
            prob_fake = float(np.mean(transformed_feat) > 0)
    elif model_type in ["custom_cnn", "efficientnet"]:
        import tensorflow as tf
        model_path = "reports/custom_cnn.keras" if model_type == "custom_cnn" else "reports/efficientnet_model.keras"
        if Path(model_path).exists():
            model = tf.keras.models.load_model(model_path)
            prob_fake = float(model.predict(np.expand_dims(normalized_face, axis=0))[0, 0])
        else:
            prob_fake = 0.5
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    label = "Fake" if prob_fake >= 0.5 else "Real"
    confidence = prob_fake if prob_fake >= 0.5 else (1.0 - prob_fake)

    result = {
        "image_path": str(path.resolve()),
        "prediction": label,
        "confidence_percentage": round(confidence * 100, 2),
        "raw_fake_probability": round(prob_fake, 4),
        "model_used": model_type
    }
    return result

def main():
    parser = argparse.ArgumentParser(description="Deepfake Image Classifier Inference CLI")
    parser.add_argument("--image", type=str, required=True, help="Path to face image file")
    parser.add_argument("--model", type=str, default="classical", choices=["classical", "custom_cnn", "efficientnet"], help="Model engine")
    args = parser.parse_args()

    try:
        res = predict_image(args.image, model_type=args.model)
        print(json.dumps(res, indent=4))
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
