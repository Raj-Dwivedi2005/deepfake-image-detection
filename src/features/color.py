"""Color and HSV histogram inconsistency feature extraction module.

Deepfake faces synthesized by neural networks frequently demonstrate subtle
chromatic aberration, HSV channel distribution variance, and inter-channel correlation discrepancies.
"""

import cv2
import numpy as np

class ColorFeatureExtractor:
    """Extracts RGB and HSV color space histogram statistics and channel correlations."""

    def __init__(self, num_bins_per_channel: int = 16):
        """Initialize color feature extractor.

        Args:
            num_bins_per_channel: Number of histogram bins per color channel.
        """
        self.num_bins = num_bins_per_channel

    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract color distribution and HSV inconsistency features.

        Args:
            image: Input RGB image array (H, W, 3) uint8 or float32 [0, 1].

        Returns:
            1D numpy array of color features.
        """
        if image.dtype != np.uint8:
            img_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        else:
            img_uint8 = image

        # 1. RGB channel histograms
        r_hist, _ = np.histogram(img_uint8[:, :, 0], bins=self.num_bins, range=(0, 256), density=True)
        g_hist, _ = np.histogram(img_uint8[:, :, 1], bins=self.num_bins, range=(0, 256), density=True)
        b_hist, _ = np.histogram(img_uint8[:, :, 2], bins=self.num_bins, range=(0, 256), density=True)

        # 2. HSV channel histograms
        hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
        h_hist, _ = np.histogram(hsv[:, :, 0], bins=self.num_bins, range=(0, 180), density=True)
        s_hist, _ = np.histogram(hsv[:, :, 1], bins=self.num_bins, range=(0, 256), density=True)
        v_hist, _ = np.histogram(hsv[:, :, 2], bins=self.num_bins, range=(0, 256), density=True)

        # 3. Inter-channel correlation (R vs G, G vs B, R vs B)
        r_flat = img_uint8[:, :, 0].ravel().astype(np.float32)
        g_flat = img_uint8[:, :, 1].ravel().astype(np.float32)
        b_flat = img_uint8[:, :, 2].ravel().astype(np.float32)

        corr_rg = float(np.corrcoef(r_flat, g_flat)[0, 1])
        corr_gb = float(np.corrcoef(g_flat, b_flat)[0, 1])
        corr_rb = float(np.corrcoef(r_flat, b_flat)[0, 1])

        features = np.concatenate([
            r_hist, g_hist, b_hist,
            h_hist, s_hist, v_hist,
            [corr_rg, corr_gb, corr_rb]
        ]).astype(np.float32)

        # Handle NaNs in correlation if image is uniform
        features = np.nan_to_num(features, nan=0.0)
        return features
