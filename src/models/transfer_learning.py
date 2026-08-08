"""Transfer learning deepfake classification model using ImageNet pretrained EfficientNetB0/ResNet50V2.

Fine-tunes high-level features for state-of-the-art binary deepfake face detection.
"""

from typing import Tuple, Optional
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0, ResNet50V2

def build_transfer_learning_model(
    backbone_name: str = "EfficientNetB0",
    input_shape: Tuple[int, int, int] = (256, 256, 3),
    learning_rate: float = 0.0005,
    fine_tune_at: Optional[int] = 100
) -> tf.keras.Model:
    """Build fine-tuned transfer learning binary classifier.

    Args:
        backbone_name: Architecture choice ('EfficientNetB0' or 'ResNet50V2').
        input_shape: Image input tuple (height, width, channels).
        learning_rate: Optimizer initial learning rate.
        fine_tune_at: Unfreeze base model layers starting from index `fine_tune_at`.

    Returns:
        Compiled Transfer Learning Keras Model.
    """
    inputs = layers.Input(shape=input_shape)

    if backbone_name == "EfficientNetB0":
        base_model = EfficientNetB0(weights="imagenet", include_top=False, input_tensor=inputs)
    elif backbone_name == "ResNet50V2":
        base_model = ResNet50V2(weights="imagenet", include_top=False, input_tensor=inputs)
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    # Freeze base model initially up to fine_tune_at
    if fine_tune_at is not None:
        base_model.trainable = True
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
    else:
        base_model.trainable = False

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=f"Transfer_{backbone_name}")

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc"), tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")]
    )

    return model
