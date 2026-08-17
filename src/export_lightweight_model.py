"""Lightweight model export script for production deployment.

Trains and exports lightweight classical Random Forest model and preprocessing artifacts
(scaler.pkl, pca.pkl, classical_model.pkl) suitable for Git tracking and Streamlit Cloud deployment.
"""

import logging
from pathlib import Path
import numpy as np

from src.data.dataset import DeepfakeDataset
from src.features.pipeline import HandcraftedFeatureExtractor
from src.models.classical import ClassicalBaselineModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("export_model")

def export_lightweight_artifacts(output_dir: str = "reports") -> None:
    """Train and save lightweight inference artifacts (< 1MB total).

    Args:
        output_dir: Directory where scaler, pca, and model pickles are saved.
    """
    logger.info("Initializing lightweight model export...")
    dataset_mgr = DeepfakeDataset(data_dir="data/raw", image_size=(256, 256), seed=42)
    raw_data_dir = dataset_mgr.generate_synthetic_fallback_dataset(samples_per_class=150)
    paths, labels = dataset_mgr.collect_image_paths(raw_data_dir, max_samples=300)

    splits = dataset_mgr.get_stratified_split(paths, labels, ratios=(0.70, 0.15, 0.15))
    train_paths, train_labels = splits["train"]

    logger.info(f"Loading and preprocessing {len(train_paths)} training images...")
    X_train_imgs = [dataset_mgr.load_and_preprocess_image(p) for p in train_paths]
    y_train = np.array(train_labels, dtype=int)

    logger.info("Extracting feature vectors...")
    feature_pipeline = HandcraftedFeatureExtractor(n_pca_components=50)
    raw_train = feature_pipeline.extract_batch_raw(X_train_imgs)
    X_train_feats = feature_pipeline.fit_transform(raw_train)

    scaler_path = str(Path(output_dir) / "scaler.pkl")
    pca_path = str(Path(output_dir) / "pca.pkl")
    model_path = str(Path(output_dir) / "classical_model.pkl")

    feature_pipeline.save(scaler_path=scaler_path, pca_path=pca_path)

    logger.info("Training Random Forest classifier...")
    model = ClassicalBaselineModel(model_type="random_forest", n_estimators=50, max_depth=10, random_state=42)
    model.train(X_train_feats, y_train)
    model.save(filepath=model_path)

    logger.info("Lightweight deployment artifacts exported successfully!")

if __name__ == "__main__":
    export_lightweight_artifacts()
