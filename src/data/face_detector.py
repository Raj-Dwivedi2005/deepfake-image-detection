"""Face detection and alignment module using OpenCV.

Provides robust face extraction with fallback mechanisms to ensure
pipeline stability when face cascades fail or images are tightly cropped.
"""

import logging
from typing import Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class FaceDetector:
    """OpenCV-based face detector with fallback bounding box extraction."""

    def __init__(self, cascade_path: Optional[str] = None):
        """Initialize OpenCV Haar Cascade Face Detector.

        Args:
            cascade_path: Path to Haar Cascade XML. Defaults to OpenCV default frontal face cascade.
        """
        if cascade_path is None:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            logger.warning(f"Failed to load Haar Cascade from {cascade_path}. Fallback mode active.")

    def detect_and_crop(self, image: np.ndarray, target_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
        """Detect primary face in image, crop, and resize to target_size.

        If no face is detected or cascade fails, falls back to center-crop / standard resize.

        Args:
            image: Input BGR or RGB image as numpy array (H, W, C).
            target_size: Target (width, height) tuple for output image.

        Returns:
            Cropped and resized face image as numpy array of shape (*target_size, C).
        """
        if image is None or image.size == 0:
            raise ValueError("Input image is empty or invalid.")

        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 and image.shape[2] == 3 else image
        
        faces = []
        if not self.face_cascade.empty():
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )

        if len(faces) > 0:
            # Pick largest detected face by area
            x, y, fw, fh = max(faces, key=lambda rect: rect[2] * rect[3])
            
            # Add subtle padding (10%) around bounding box
            pad_w, pad_h = int(fw * 0.1), int(fh * 0.1)
            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(w, x + fw + pad_w)
            y2 = min(h, y + fh + pad_h)

            cropped = image[y1:y2, x1:x2]
        else:
            # Fallback: Center crop preserving square aspect ratio
            min_dim = min(h, w)
            start_x = (w - min_dim) // 2
            start_y = (h - min_dim) // 2
            cropped = image[start_y : start_y + min_dim, start_x : start_x + min_dim]

        resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)
        return resized
