"""High-pass filter noise-residual statistical feature extraction module.

Natural camera sensor noise (PRNU, shot noise) differs significantly from synthetic
generator noise residual signatures. High-pass residual statistical moments expose this gap.
"""

import cv2
import numpy as np

class NoiseResidualFeatureExtractor:
    """Extracts statistical metrics from high-pass noise residuals using fast vector NumPy."""

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

        # Fast NumPy statistical moment calculations
        res_mean = float(np.mean(residual))
        res_var = float(np.var(residual))
        std_res = np.sqrt(res_var) + 1e-8
        res_skew = float(np.mean(((residual - res_mean) / std_res) ** 3))
        res_kurt = float(np.mean(((residual - res_mean) / std_res) ** 4) - 3.0)

        lap_mean = float(np.mean(laplacian_res))
        lap_var = float(np.var(laplacian_res))
        std_lap = np.sqrt(lap_var) + 1e-8
        lap_skew = float(np.mean(((laplacian_res - lap_mean) / std_lap) ** 3))
        lap_kurt = float(np.mean(((laplacian_res - lap_mean) / std_lap) ** 4) - 3.0)

        return np.array(
            [res_mean, res_var, res_skew, res_kurt, lap_mean, lap_var, lap_skew, lap_kurt],
            dtype=np.float32
        )
