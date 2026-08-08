"""High-pass filter noise-residual statistical feature extraction module.

Natural camera sensor noise (PRNU, shot noise) differs significantly from synthetic
generator noise residual signatures. High-pass residual statistical moments expose this gap.
"""

import cv2
import numpy as np
from scipy.stats import skew, kurtosis

class NoiseResidualFeatureExtractor:
    """Extracts statistical metrics from high-pass noise residuals."""

    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract high-pass noise residual statistical moments.

        Args:
            image: Input image array uint8 [0, 255] or float32 [0, 1].

        Returns:
            1D numpy array of residual statistical metrics (mean, var, skewness, kurtosis).
        """
        if image.dtype != np.uint8:
            img_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        else:
            img_uint8 = image

        if img_uint8.ndim == 3:
            gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY).astype(np.float32)
        else:
            gray = img_uint8.astype(np.float32)

        # High-pass filter residual via Gaussian subtraction (Denoising residual)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        residual = gray - blurred

        # Laplacian high-pass filter residual
        laplacian_res = cv2.Laplacian(gray, cv2.CV_32F)

        # Compute moments for Gaussian residual
        res_mean = np.mean(residual)
        res_var = np.var(residual)
        res_skew = float(skew(residual.ravel()))
        res_kurt = float(kurtosis(residual.ravel()))

        # Compute moments for Laplacian residual
        lap_mean = np.mean(laplacian_res)
        lap_var = np.var(laplacian_res)
        lap_skew = float(skew(laplacian_res.ravel()))
        lap_kurt = float(kurtosis(laplacian_res.ravel()))

        return np.array(
            [res_mean, res_var, res_skew, res_kurt, lap_mean, lap_var, lap_skew, lap_kurt],
            dtype=np.float32
        )
