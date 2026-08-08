"""Visualization utilities module.

Generates publication-ready figures for ROC curves, confusion matrices,
model comparison charts, and Grad-CAM sample visualizations.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix

logger = logging.getLogger(__name__)

class Visualizer:
    """Generates and exports diagnostic plots to reports/figures/."""

    def __init__(self, output_dir: str = "reports/figures"):
        """Initialize visualizer.

        Args:
            output_dir: Folder to store rendered plots.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid", palette="muted")

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> str:
        """Plot and save confusion matrix heatmap.

        Args:
            y_true: True binary ground-truth labels.
            y_pred: Binary predicted labels.
            model_name: Name identifier of model.

        Returns:
            Saved figure path string.
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["Real (0)", "Fake (1)"],
                    yticklabels=["Real (0)", "Fake (1)"])
        plt.title(f"Confusion Matrix - {model_name}", fontsize=14, fontweight="bold")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()

        filepath = self.output_dir / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
        plt.savefig(filepath, dpi=300)
        plt.close()
        logger.info(f"Saved confusion matrix plot to {filepath}")
        return str(filepath)

    def plot_roc_curves(self, roc_data: Dict[str, Dict[str, np.ndarray]]) -> str:
        """Plot multi-model ROC curves on single canvas.

        Args:
            roc_data: Dict mapping model names to {'y_true': ..., 'y_prob': ...}.

        Returns:
            Saved figure path string.
        """
        plt.figure(figsize=(8, 6))

        for model_name, data in roc_data.items():
            fpr, tpr, _ = roc_curve(data["y_true"], data["y_prob"])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC = {roc_auc:.3f})")

        plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Chance (0.50)")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontsize=12)
        plt.ylabel("True Positive Rate", fontsize=12)
        plt.title("ROC Curves - Deepfake Classification Comparison", fontsize=14, fontweight="bold")
        plt.legend(loc="lower right", fontsize=10)
        plt.tight_layout()

        filepath = self.output_dir / "roc_curves_comparison.png"
        plt.savefig(filepath, dpi=300)
        plt.close()
        logger.info(f"Saved ROC curves plot to {filepath}")
        return str(filepath)

    def plot_model_comparison_bar(self, metrics_summary: Dict[str, Dict[str, float]]) -> str:
        """Plot bar chart comparing models across Accuracy, F1, and ROC-AUC.

        Args:
            metrics_summary: Dict mapping model names to metric dictionaries.

        Returns:
            Saved figure path string.
        """
        models_list = list(metrics_summary.keys())
        accuracies = [metrics_summary[m]["accuracy"] for m in models_list]
        f1_scores = [metrics_summary[m]["f1_score"] for m in models_list]
        aucs = [metrics_summary[m]["roc_auc"] for m in models_list]

        x = np.arange(len(models_list))
        width = 0.25

        plt.figure(figsize=(10, 6))
        plt.bar(x - width, accuracies, width, label="Accuracy", color="#3498db")
        plt.bar(x, f1_scores, width, label="F1-Score", color="#2ecc71")
        plt.bar(x + width, aucs, width, label="ROC-AUC", color="#e74c3c")

        plt.xlabel("Model Architecture", fontsize=12, fontweight="bold")
        plt.ylabel("Score", fontsize=12, fontweight="bold")
        plt.title("Model Benchmarking Comparison", fontsize=14, fontweight="bold")
        plt.xticks(x, models_list, rotation=15, ha="right")
        plt.ylim([0.0, 1.05])
        plt.legend(loc="lower right")
        plt.tight_layout()

        filepath = self.output_dir / "model_comparison_bar.png"
        plt.savefig(filepath, dpi=300)
        plt.close()
        logger.info(f"Saved model comparison bar chart to {filepath}")
        return str(filepath)

    def save_gradcam_panel(
        self,
        orig_img: np.ndarray,
        heatmap_rgb: np.ndarray,
        overlay_rgb: np.ndarray,
        pred_label: str,
        confidence: float,
        filename: str = "gradcam_sample.png"
    ) -> str:
        """Save a 3-panel figure showing Original Image, Grad-CAM Heatmap, and Overlay.

        Args:
            orig_img: Original face image RGB uint8 or float.
            heatmap_rgb: Colorized Grad-CAM heatmap.
            overlay_rgb: Blended overlay image.
            pred_label: Predicted label string ('Real' or 'Fake').
            confidence: Prediction confidence score [0, 1].
            filename: Target output image filename.

        Returns:
            Saved figure path string.
        """
        if orig_img.dtype != np.uint8:
            orig_img = np.clip(orig_img * 255.0, 0, 255).astype(np.uint8)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        ax1.imshow(orig_img)
        ax1.set_title("Input Face Image", fontsize=12, fontweight="bold")
        ax1.axis("off")

        ax2.imshow(heatmap_rgb)
        ax2.set_title("Grad-CAM Activation Map", fontsize=12, fontweight="bold")
        ax2.axis("off")

        ax3.imshow(overlay_rgb)
        ax3.set_title(f"Overlay ({pred_label}: {confidence*100:.1f}%)", fontsize=12, fontweight="bold")
        ax3.axis("off")

        plt.tight_layout()
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300)
        plt.close()
        logger.info(f"Saved Grad-CAM panel figure to {filepath}")
        return str(filepath)
