"""Unit tests for src/models module."""

import pytest
import numpy as np
from src.models.classical import ClassicalBaselineModel
from src.models.custom_cnn import build_custom_cnn
from src.models.fusion import build_fusion_model

def test_classical_model_fit_predict():
    model = ClassicalBaselineModel(model_type="random_forest", n_estimators=10, random_state=42)
    X = np.random.rand(30, 20)
    y = np.array([0]*15 + [1]*15)
    model.train(X, y)
    probs = model.predict_proba(X)
    assert probs.shape == (30, 2)
    assert ((probs >= 0.0) & (probs <= 1.0)).all()

def test_custom_cnn_output_shape():
    cnn = build_custom_cnn(input_shape=(128, 128, 3))
    dummy_input = np.random.rand(2, 128, 128, 3).astype(np.float32)
    output = cnn.predict(dummy_input)
    assert output.shape == (2, 1)
    assert ((output >= 0.0) & (output <= 1.0)).all()

def test_fusion_model_output_shape():
    fusion = build_fusion_model(image_shape=(128, 128, 3), feature_dim=15)
    dummy_img = np.random.rand(2, 128, 128, 3).astype(np.float32)
    dummy_feat = np.random.rand(2, 15).astype(np.float32)
    output = fusion.predict({"image_input": dummy_img, "feature_input": dummy_feat})
    assert output.shape == (2, 1)
    assert ((output >= 0.0) & (output <= 1.0)).all()
