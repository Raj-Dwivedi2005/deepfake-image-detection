"""Unit tests for src/data module."""

import pytest
import numpy as np
from src.data.face_detector import FaceDetector
from src.data.augmentation import ImageAugmenter
from src.data.dataset import DeepfakeDataset

def test_face_detector_crop_shape():
    detector = FaceDetector()
    dummy_img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    cropped = detector.detect_and_crop(dummy_img, target_size=(256, 256))
    assert cropped.shape == (256, 256, 3)

def test_jpeg_augmentation():
    augmenter = ImageAugmenter(jpeg_quality_range=(40, 50))
    dummy_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    augmented = augmenter.augment(dummy_img, apply_jpeg=True)
    assert augmented.shape == (256, 256, 3)
    assert augmented.dtype == np.uint8

def test_stratified_split():
    dataset_mgr = DeepfakeDataset(seed=42)
    paths = [f"img_{i}.jpg" for i in range(100)]
    labels = [0] * 50 + [1] * 50
    splits = dataset_mgr.get_stratified_split(paths, labels, ratios=(0.70, 0.15, 0.15))

    train_paths, train_labels = splits["train"]
    val_paths, val_labels = splits["val"]
    test_paths, test_labels = splits["test"]

    assert len(train_paths) == 70
    assert len(val_paths) == 15
    assert len(test_paths) == 15
    assert sum(train_labels) == 35  # Stratified balanced class distribution
