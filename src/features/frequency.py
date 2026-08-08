"""Frequency-domain feature extraction using 2D FFT and 2D DCT.

GAN upsampling layers (e.g. transposed convolutions in StyleGAN) leave periodic
spectral artifacts and high-frequency grid anomalies in the Fourier and DCT domains.
"""

from typing import Tuple
import cv2
import numpy as np

class FrequencyFeatureExtractor:
    """Extracts 2D Fast Fourier Transform (FFT) and Discrete Cosine Transform (DCT) features."""

    def __init__(self, num_radial_bins: int = 16):
        """Initialize frequency feature extractor.

        Args:
            num_radial_bins: Number of concentric radial bins for power spectrum aggregation.
        """
        self.num_radial_bins = num_radial_bins

    def extract_fft_features(self, gray_img: np.ndarray) -> np.ndarray:
        """Extract 2D FFT magnitude spectrum statistics and radial energy profile.

        Args:
            gray_img: Grayscale float32 image array scaled [0, 1] or uint8 [0, 255].

        Returns:
            1D numpy array of FFT spectral features.
        """
        h, w = gray_img.shape
        # Compute 2D FFT and shift zero frequency to center
        fft = np.fft.fft2(gray_img)
        fft_shift = np.fft.fftshift(fft)
        magnitude_spectrum = np.log(np.abs(fft_shift) + 1e-8)

        # Global statistics
        mean_mag = np.mean(magnitude_spectrum)
        std_mag = np.std(magnitude_spectrum)
        max_mag = np.max(magnitude_spectrum)

        # Concentric Radial Spectrum Profile
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

        max_radius = np.sqrt(cx**2 + cy**2)
        bin_edges = np.linspace(0, max_radius, self.num_radial_bins + 1)
        radial_energy = []

        for i in range(self.num_radial_bins):
            mask = (r >= bin_edges[i]) & (r < bin_edges[i + 1])
            if np.any(mask):
                radial_energy.append(np.mean(magnitude_spectrum[mask]))
            else:
                radial_energy.append(0.0)

        radial_profile = np.array(radial_energy, dtype=np.float32)

        # High-frequency ratio (outer quarter of spectrum)
        high_freq_mask = r > (max_radius * 0.75)
        high_freq_ratio = np.sum(magnitude_spectrum[high_freq_mask]) / (np.sum(magnitude_spectrum) + 1e-8)

        return np.concatenate(([mean_mag, std_mag, max_mag, high_freq_ratio], radial_profile))

    def extract_dct_features(self, gray_img: np.ndarray) -> np.ndarray:
        """Extract Discrete Cosine Transform (DCT) coefficient statistics.

        Args:
            gray_img: Grayscale image array (float32 [0, 1]).

        Returns:
            1D numpy array of DCT coefficient summary statistics.
        """
        # 2D DCT using OpenCV
        dct = cv2.dct(gray_img.astype(np.float32))
        abs_dct = np.abs(dct)

        # Statistical properties of AC coefficients (excluding DC at 0,0)
        ac_coeffs = abs_dct.copy()
        ac_coeffs[0, 0] = 0.0

        mean_ac = np.mean(ac_coeffs)
        std_ac = np.std(ac_coeffs)
        energy = np.sum(ac_coeffs ** 2)
        high_freq_dct_energy = np.sum(ac_coeffs[gray_img.shape[0] // 2 :, gray_img.shape[1] // 2 :] ** 2)

        return np.array([mean_ac, std_ac, energy, high_freq_dct_energy], dtype=np.float32)

    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract combined FFT and DCT spectral feature vector from RGB/Grayscale image.

        Args:
            image: Input image array (float32 [0, 1] or uint8 [0, 255]).

        Returns:
            1D float32 numpy array of spectral features.
        """
        if image.ndim == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8) if image.dtype != np.uint8 else image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        else:
            gray = image.astype(np.float32) / 255.0 if image.dtype == np.uint8 else image

        fft_feats = self.extract_fft_features(gray)
        dct_feats = self.extract_dct_features(gray)
        return np.concatenate([fft_feats, dct_feats])
