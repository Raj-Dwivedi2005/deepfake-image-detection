"""Classical machine learning baseline module using Scikit-Learn.

Trains a Random Forest / SVM classifier on handcrafted spectral, texture,
noise, and color feature vectors.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

logger = logging.getLogger(__name__)

class ClassicalBaselineModel:
    """Random Forest / SVM baseline classifier trained on handcrafted feature vectors."""

    def __init__(self, model_type: str = "random_forest", n_estimators: int = 100, max_depth: int = 15, random_state: int = 42):
        """Initialize classical ML baseline model.

        Args:
            model_type: Classifier type ('random_forest' or 'svm').
            n_estimators: Number of decision trees for Random Forest.
            max_depth: Maximum tree depth for Random Forest.
            random_state: Fixed seed for reproducibility.
        """
        self.model_type = model_type
        self.random_state = random_state

        if model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=-1
            )
        elif model_type == "svm":
            self.model = SVC(probability=True, kernel="rbf", C=1.0, random_state=random_state)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        self.is_trained = False

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit classical model on training features.

        Args:
            X_train: Training feature matrix (N_samples, N_features).
            y_train: Training labels array (N_samples,).
        """
        logger.info(f"Training {self.model_type} baseline on feature matrix shape {X_train.shape}...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        logger.info("Classical baseline model training complete.")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for input features.

        Args:
            X: Input feature matrix (N_samples, N_features).

        Returns:
            2D numpy array of probabilities [P(Real), P(Fake)].
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet.")
        return self.model.predict_proba(X)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate baseline model performance on test set.

        Args:
            X_test: Test set feature matrix.
            y_test: Test set labels.

        Returns:
            Dictionary of metrics (accuracy, precision, recall, f1, roc_auc).
        """
        y_prob = self.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_prob)),
        }
        logger.info(f"Classical Model ({self.model_type}) Evaluation -> {metrics}")
        return metrics

    def save(self, filepath: str = "reports/classical_model.pkl") -> None:
        """Save trained model to disk.

        Args:
            filepath: Target file path for pickle output.
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self.model, f)
        logger.info(f"Saved classical model to {filepath}")

    def load(self, filepath: str = "reports/classical_model.pkl") -> None:
        """Load trained model from disk.

        Args:
            filepath: Path to pickled model file.
        """
        with open(filepath, "rb") as f:
            self.model = pickle.load(f)
        self.is_trained = True
        logger.info(f"Loaded classical model from {filepath}")
