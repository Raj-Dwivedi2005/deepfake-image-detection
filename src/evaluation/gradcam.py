"""Gradient-weighted Class Activation Mapping (Grad-CAM) module.

Generates visual heatmaps highlighting deepfake features and artifacts
that influenced CNN / Transfer Learning model classification decisions.
"""

import logging
from typing import Optional, Tuple
import cv2
import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)

class GradCAM:
    """Computes Grad-CAM activation heatmaps for Keras convolutional neural networks."""

    def __init__(self, model: tf.keras.Model, layer_name: Optional[str] = None):
        """Initialize Grad-CAM manager.

        Args:
            model: Trained Keras model.
            layer_name: Target conv layer name. If None, automatically locates last Conv2D layer.
        """
        self.model = model
        self.layer_name = layer_name or self._find_last_conv_layer()
        logger.info(f"Initialized Grad-CAM targeting layer: '{self.layer_name}'")

    def _find_last_conv_layer(self) -> str:
        """Find the name of the final Conv2D layer in model architecture.

        Returns:
            Layer name string.
        """
        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer.name
            # Search inside nested functional base models (e.g., EfficientNet)
            if hasattr(layer, "layers"):
                for sub_layer in reversed(layer.layers):
                    if isinstance(sub_layer, tf.keras.layers.Conv2D):
                        return sub_layer.name
        raise ValueError("No Conv2D layer found in the model architecture for Grad-CAM.")

    def compute_heatmap(self, image_tensor: np.ndarray) -> np.ndarray:
        """Compute Grad-CAM activation heatmap matrix for input image tensor.

        Args:
            image_tensor: Batch image array of shape (1, H, W, 3) float32 [0, 1].

        Returns:
            2D numpy array heatmap scaled [0, 1] of shape (H, W).
        """
        # Create gradient model returning target layer activations and model predictions
        try:
            target_layer = self.model.get_layer(self.layer_name)
            grad_model = tf.keras.models.Model(
                inputs=self.model.inputs,
                outputs=[target_layer.output, self.model.output]
            )
        except ValueError:
            # Handle nested submodel case (e.g. EfficientNet inside Functional model)
            for layer in self.model.layers:
                if hasattr(layer, "get_layer"):
                    try:
                        target_layer = layer.get_layer(self.layer_name)
                        sub_grad_model = tf.keras.models.Model(
                            inputs=layer.inputs,
                            outputs=target_layer.output
                        )
                        # Fallback simple gradient computation
                        break
                    except ValueError:
                        continue
            grad_model = tf.keras.models.Model(
                inputs=self.model.inputs,
                outputs=[self.model.layers[-2].output, self.model.output]
            )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image_tensor)
            loss = predictions[0]

        # Calculate gradients of top predicted class w.r.t conv_outputs
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # Relu on heatmap and normalize
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()

    def overlay_heatmap(
        self,
        original_image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Overlay Grad-CAM heatmap on top of original face image.

        Args:
            original_image: Original RGB image array (H, W, 3) uint8 or float32 [0, 1].
            heatmap: 2D float heatmap array.
            alpha: Transparency factor for overlay.
            colormap: OpenCV colormap enum.

        Returns:
            Tuple of (colored_heatmap_rgb, blended_overlay_rgb).
        """
        if original_image.dtype != np.uint8:
            img_uint8 = np.clip(original_image * 255.0, 0, 255).astype(np.uint8)
        else:
            img_uint8 = original_image

        h, w = img_uint8.shape[:2]
        resized_heatmap = cv2.resize(heatmap, (w, h))
        heatmap_uint8 = np.uint8(255 * resized_heatmap)

        heatmap_colored_bgr = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_colored_rgb = cv2.cvtColor(heatmap_colored_bgr, cv2.COLOR_BGR2RGB)

        blended = cv2.addWeighted(img_uint8, 1 - alpha, heatmap_colored_rgb, alpha, 0)
        return heatmap_colored_rgb, blended
