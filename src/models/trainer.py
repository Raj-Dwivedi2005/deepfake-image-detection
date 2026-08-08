"""Training manager module for Keras deep learning models.

Manages training callbacks, checkpointing, evaluation, and training curve figures generation.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
import tensorflow as tf

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Manages deep learning model training, callbacks, and visualization."""

    def __init__(self, model: tf.keras.Model, model_save_path: str = "reports/model.keras"):
        """Initialize ModelTrainer.

        Args:
            model: Compiled Keras model instance.
            model_save_path: Path to save best checkpoint.
        """
        self.model = model
        self.model_save_path = Path(model_save_path)
        self.model_save_path.parent.mkdir(parents=True, exist_ok=True)
        self.history: Optional[tf.keras.callbacks.History] = None

    def get_callbacks(self, patience: int = 5) -> List[tf.keras.callbacks.Callback]:
        """Configure standard Keras callbacks.

        Args:
            patience: Number of epochs to wait for improvement before early stopping / LR reduction.

        Returns:
            List of Keras Callback objects.
        """
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath=str(self.model_save_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )
        early_stopping_cb = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(2, patience // 2),
            min_lr=1e-6,
            verbose=1
        )
        return [checkpoint_cb, early_stopping_cb, reduce_lr_cb]

    def train(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        epochs: int = 15,
        patience: int = 5
    ) -> tf.keras.callbacks.History:
        """Train Keras model using tf.data.Dataset pipelines.

        Args:
            train_dataset: Training tf.data.Dataset.
            val_dataset: Validation tf.data.Dataset.
            epochs: Maximum training epochs.
            patience: Early stopping patience.

        Returns:
            Training History object.
        """
        logger.info(f"Starting training for {self.model.name} for up to {epochs} epochs...")
        callbacks = self.get_callbacks(patience=patience)
        self.history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks
        )
        logger.info(f"Training completed for {self.model.name}.")
        return self.history

    def plot_training_curves(self, output_fig_path: str = "reports/figures/training_curves.png") -> None:
        """Save loss and accuracy per epoch figures.

        Args:
            output_fig_path: Destination path for saving plot PNG.
        """
        if self.history is None:
            logger.warning("No training history available to plot.")
            return

        Path(output_fig_path).parent.mkdir(parents=True, exist_ok=True)
        hist = self.history.history

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Loss plot
        ax1.plot(hist["loss"], label="Train Loss", color="#1f77b4", linewidth=2)
        if "val_loss" in hist:
            ax1.plot(hist["val_loss"], label="Val Loss", color="#ff7f0e", linewidth=2, linestyle="--")
        ax1.set_title(f"{self.model.name} - Loss per Epoch")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy plot
        ax2.plot(hist["accuracy"], label="Train Accuracy", color="#2ca02c", linewidth=2)
        if "val_accuracy" in hist:
            ax2.plot(hist["val_accuracy"], label="Val Accuracy", color="#d62728", linewidth=2, linestyle="--")
        ax2.set_title(f"{self.model.name} - Accuracy per Epoch")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_fig_path, dpi=300)
        plt.close()
        logger.info(f"Training curves figure saved to {output_fig_path}")
