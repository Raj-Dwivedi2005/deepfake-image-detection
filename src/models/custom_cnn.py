"""Custom CNN baseline model built from scratch in Keras/TensorFlow.

Provides a lightweight deep learning architecture for binary face classification.
"""

from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers, models

def build_custom_cnn(
    input_shape: Tuple[int, int, int] = (256, 256, 3),
    learning_rate: float = 0.001
) -> tf.keras.Model:
    """Build and compile a custom convolutional neural network from scratch.

    Args:
        input_shape: Image input dimensions (height, width, channels).
        learning_rate: Initial Adam optimizer learning rate.

    Returns:
        Compiled Keras Model object.
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),

        # Block 1
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # Block 3
        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),

        # Dense Classifier Head
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid")
    ], name="Custom_CNN_Scratch")

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc"), tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")]
    )

    return model
