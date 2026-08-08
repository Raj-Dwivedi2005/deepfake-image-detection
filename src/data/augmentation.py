"""Data augmentation pipeline for deepfake image classification.

Includes domain-specific JPEG re-compression augmentation to expose or augment
compression artifacts, alongside standard spatial and color transformations.
"""

import random
from typing import Tuple
import cv2
import numpy as np

class ImageAugmenter:
    """Augmentation pipeline with deepfake-specific JPEG re-compression."""

    def __init__(
        self,
        jpeg_quality_range: Tuple[int, int] = (40, 85),
        rotation_range: int = 15,
        brightness_range: Tuple[float, float] = (0.8, 1.2),
        flip_horizontal: bool = True,
    ):
        """Initialize data augmenter.

        Args:
            jpeg_quality_range: Range of JPEG compression quality factors (1-100).
            rotation_range: Maximum rotation angle in degrees.
            brightness_range: Range for random brightness scaling.
            flip_horizontal: Whether to randomly flip horizontally.
        """
        self.jpeg_quality_range = jpeg_quality_range
        self.rotation_range = rotation_range
        self.brightness_range = brightness_range
        self.flip_horizontal = flip_horizontal

    def apply_jpeg_compression(self, image: np.ndarray, quality: int) -> np.ndarray:
        """Simulate JPEG compression artifacts at specified quality factor.

        Args:
            image: Input image uint8 array (H, W, 3).
            quality: JPEG compression quality factor between 1 and 100.

        Returns:
            Re-compressed image uint8 array.
        """
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        result, enc_img = cv2.imencode('.jpg', image, encode_param)
        if not result:
            return image
        dec_img = cv2.imdecode(enc_img, cv2.IMREAD_COLOR)
        return dec_img

    def augment(self, image: np.ndarray, apply_jpeg: bool = True) -> np.ndarray:
        """Apply random data augmentations to an image.

        Args:
            image: Input uint8 image array (H, W, C).
            apply_jpeg: Whether to apply random JPEG re-compression.

        Returns:
            Augmented image array (H, W, C).
        """
        aug_img = image.copy()

        # 1. Random Horizontal Flip
        if self.flip_horizontal and random.random() > 0.5:
            aug_img = cv2.flip(aug_img, 1)

        # 2. Random Small Rotation
        if self.rotation_range > 0 and random.random() > 0.5:
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            h, w = aug_img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            aug_img = cv2.warpAffine(aug_img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        # 3. Brightness / Contrast Jitter
        if random.random() > 0.5:
            alpha = random.uniform(self.brightness_range[0], self.brightness_range[1])
            beta = random.randint(-15, 15)
            aug_img = cv2.convertScaleAbs(aug_img, alpha=alpha, beta=beta)

        # 4. JPEG Re-compression Augmentation (Deepfake cue targeting)
        if apply_jpeg and random.random() > 0.4:
            quality = random.randint(self.jpeg_quality_range[0], self.jpeg_quality_range[1])
            aug_img = self.apply_jpeg_compression(aug_img, quality)

        return aug_img
