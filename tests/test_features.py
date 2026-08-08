"""Unit tests for src/features module."""

import pytest
import numpy as np
from src.features.frequency import FrequencyFeatureExtractor
from src.features.texture import TextureFeatureExtractor
from src.features.noise import NoiseResidualFeatureExtractor
from src.features.color import ColorFeatureExtractor
from src.features.pipeline import HandcraftedFeatureExtractor

def test_frequency_extractor():
    extractor = FrequencyFeatureExtractor(num_radial_bins=16)
    dummy_img = np.random.rand(256, 256, 3).astype(np.float32)
    feats = extractor.extract(dummy_img)
    assert feats.ndim == 1
    assert not np.isnan(feats).any()
    assert len(feats) > 10

def test_texture_extractor():
    extractor = TextureFeatureExtractor(P=24, R=3, method="uniform")
    dummy_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    feats = extractor.extract(dummy_img)
    assert len(feats) == 26
    assert not np.isnan(feats).any()

def test_noise_extractor():
    extractor = NoiseResidualFeatureExtractor()
    dummy_img = np.random.rand(256, 256, 3).astype(np.float32)
    feats = extractor.extract(dummy_img)
    assert len(feats) == 8
    assert not np.isnan(feats).any()

def test_color_extractor():
    extractor = ColorFeatureExtractor(num_bins_per_channel=16)
    dummy_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    feats = extractor.extract(dummy_img)
    assert len(feats) == (16 * 6 + 3)
    assert not np.isnan(feats).any()

def test_pipeline_pca_transform():
    pipeline = HandcraftedFeatureExtractor(n_pca_components=10)
    dummy_batch = [np.random.rand(256, 256, 3).astype(np.float32) for _ in range(15)]
    raw = pipeline.extract_batch_raw(dummy_batch)
    transformed = pipeline.fit_transform(raw)
    assert transformed.shape == (15, 10)
    assert not np.isnan(transformed).any()
