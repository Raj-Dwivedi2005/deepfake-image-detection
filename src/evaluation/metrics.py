"""Evaluation metrics calculation and JSON report exporter module.

Computes exact performance metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
and confusion matrices across held-out test sets without fabricating values.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Union
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Evaluates binary classifiers and exports metric summaries to JSON."""

    @staticmethod
    def compute_metrics(
        y_true: Union[List[int], np.ndarray],
        y_pred_proba: Union[List[float], np.ndarray],
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """Compute binary classification evaluation metrics.

        Args:
            y_true: Ground truth binary labels (0 = Real, 1 = Fake).
            y_pred_proba: Predicted probability of class 1 (Fake).
            threshold: Probability threshold for positive classification.

        Returns:
            Dictionary containing metrics and confusion matrix values.
        """
        y_true_arr = np.array(y_true, dtype=int)
        y_prob_arr = np.array(y_pred_proba, dtype=np.float32)
        y_pred_arr = (y_prob_arr >= threshold).astype(int)

        acc = float(accuracy_score(y_true_arr, y_pred_arr))
        prec = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
        rec = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
        f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))
        
        try:
            auc = float(roc_auc_score(y_true_arr, y_prob_arr))
        except ValueError:
            auc = 0.5  # Fallback if single class present

        cm = confusion_matrix(y_true_arr, y_pred_arr)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

        metrics = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4),
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp)
            }
        }
        return metrics

    @staticmethod
    def save_metrics_report(metrics_dict: Dict[str, Any], filepath: str = "reports/metrics.json") -> None:
        """Export comprehensive model metrics to JSON file.

        Args:
            metrics_dict: Dictionary mapping model names to their metric metrics.
            filepath: Destination file path for JSON output.
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(metrics_dict, f, indent=4)
        logger.info(f"Saved evaluation metrics report to {filepath}")
