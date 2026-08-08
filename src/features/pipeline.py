"""Handcrafted feature engineering pipeline combining spectral, texture, noise, and color signals.

Integrates scikit-learn StandardScaler and PCA for normalization and dimensionality reduction.
"""

import logging
import pickle
from pathlib import Path
from typing import List, Optional
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from src.features.frequency import FrequencyFeatureExtractor
from src.features.texture import TextureFeatureExtractor
from src.features.noise import NoiseResidualFeatureExtractor
from src.features.color import ColorFeatureExtractor

logger = logging.getLogger(__name__)

class HandcraftedFeatureExtractor:
    """Unified handcrafted feature extractor pipeline with StandardScaler and PCA."""

    def __init__(self, n_pca_components: int = 50):
        """Initialize feature extraction pipeline.

        Args:
            n_pca_components: Target dimensions for PCA reduction.
        """
        self.freq_extractor = FrequencyFeatureExtractor()
        self.texture_extractor = TextureFeatureExtractor()
        self.noise_extractor = NoiseResidualFeatureExtractor()
        self.color_extractor = ColorFeatureExtractor()

        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_pca_components, random_state=42)
        self.is_fitted = False

    def extract_raw_features(self, image: np.ndarray) -> np.ndarray:
        """Extract unscaled concatenated feature vector for a single image.

        Args:
            image: RGB image array (H, W, 3) float32 [0, 1] or uint8 [0, 255].

        Returns:
            1D numpy feature vector.
        """
        freq_vec = self.freq_extractor.extract(image)
        texture_vec = self.texture_extractor.extract(image)
        noise_vec = self.noise_extractor.extract(image)
        color_vec = self.color_extractor.extract(image)

        raw_vec = np.concatenate([freq_vec, texture_vec, noise_vec, color_vec]).astype(np.float32)
        return raw_vec

    def extract_batch_raw(self, image_list: List[np.ndarray]) -> np.ndarray:
        """Extract raw feature matrix for a list of images.

        Args:
            image_list: List of image numpy arrays.

        Returns:
            2D numpy array of shape (N_samples, N_raw_features).
        """
        feats = [self.extract_raw_features(img) for img in image_list]
        return np.vstack(feats)

    def fit_transform(self, raw_features: np.ndarray) -> np.ndarray:
        """Fit StandardScaler and PCA on raw training features and transform.

        Args:
            raw_features: 2D numpy array (N_samples, N_raw_features).

        Returns:
            2D numpy array of transformed features (N_samples, n_pca_components).
        """
        scaled = self.scaler.fit_transform(raw_features)
        transformed = self.pca.fit_transform(scaled)
        self.is_fitted = True
        logger.info(f"Fitted Feature Extractor -> PCA Explained Variance Ratio: {np.sum(self.pca.explained_variance_ratio_):.4f}")
        return transformed

    def transform(self, raw_features: np.ndarray) -> np.ndarray:
        """Transform raw features using fitted StandardScaler and PCA.

        Args:
            raw_features: 2D numpy array (N_samples, N_raw_features).

        Returns:
            2D numpy array of transformed features.
        """
        if not self.is_fitted:
            raise ValueError("FeatureExtractor scaler and PCA are not fitted yet.")
        scaled = self.scaler.transform(raw_features)
        return self.pca.transform(scaled)

    def save(self, scaler_path: str = "reports/scaler.pkl", pca_path: str = "reports/pca.pkl") -> None:
        """Save scaler and PCA models to disk.

        Args:
            scaler_path: Output path for pickled StandardScaler.
            pca_path: Output path for pickled PCA.
        """
        Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
        with open(scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)
        with open(pca_path, "wb") as f:
            pickle.dump(self.pca, f)
        logger.info(f"Saved scaler to {scaler_path} and PCA to {pca_path}")

    def load(self, scaler_path: str = "reports/scaler.pkl", pca_path: str = "reports/pca.pkl") -> None:
        """Load scaler and PCA models from disk.

        Args:
            scaler_path: Path to pickled StandardScaler.
            pca_path: Path to pickled PCA.
        """
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        with open(pca_path, "rb") as f:
            self.pca = pickle.load(f)
        self.is_fitted = True
        logger.info(f"Loaded scaler from {scaler_path} and PCA from {pca_path}")
