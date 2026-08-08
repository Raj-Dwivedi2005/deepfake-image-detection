"""Late-fusion deepfake classification architecture.

Combines high-level spatial deep features (CNN bottleneck) with handcrafted
spectral/texture/noise feature vectors before final classification layers.
"""

from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers, models

def build_fusion_model(
    image_shape: Tuple[int, int, int] = (256, 256, 3),
    feature_dim: int = 50,
    learning_rate: float = 0.0003
) -> tf.keras.Model:
    """Build multi-input late fusion network combining image inputs and handcrafted feature vectors.

    Args:
        image_shape: Input image dimensions tuple (H, W, 3).
        feature_dim: Dimension of handcrafted feature vector (e.g. post PCA).
        learning_rate: Adam optimizer learning rate.

    Returns:
        Compiled Keras multi-input Model.
    """
    # 1. Image Stream (CNN Feature Extractor)
    image_input = layers.Input(shape=image_shape, name="image_input")
    x_img = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(image_input)
    x_img = layers.BatchNormalization()(x_img)
    x_img = layers.MaxPooling2D((2, 2))(x_img)

    x_img = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x_img)
    x_img = layers.BatchNormalization()(x_img)
    x_img = layers.MaxPooling2D((2, 2))(x_img)

    x_img = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x_img)
    x_img = layers.GlobalAveragePooling2D()(x_img)

    # 2. Handcrafted Feature Vector Stream
    feat_input = layers.Input(shape=(feature_dim,), name="feature_input")
    x_feat = layers.Dense(64, activation="relu")(feat_input)
    x_feat = layers.BatchNormalization()(x_feat)

    # 3. Concatenation / Late Fusion
    combined = layers.concatenate([x_img, x_feat], name="fusion_concatenation")

    # 4. Joint Classification Head
    z = layers.Dense(128, activation="relu")(combined)
    z = layers.Dropout(0.4)(z)
    z = layers.Dense(64, activation="relu")(z)
    z = layers.Dropout(0.3)(z)
    output = layers.Dense(1, activation="sigmoid", name="predictions")(z)

    model = models.Model(inputs=[image_input, feat_input], outputs=output, name="Late_Fusion_Model")

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc"), tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")]
    )

    return model
