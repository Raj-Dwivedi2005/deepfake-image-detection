"""Master pipeline runner script for training, evaluating, and generating diagnostic figures.

Runs full training pipeline for Classical Random Forest baseline, Custom CNN, EfficientNetB0,
and Late Fusion architectures, outputting reports/metrics.json and figures to reports/figures/.
"""

import logging
import json
import numpy as np
import tensorflow as tf
from pathlib import Path

from src.data.dataset import DeepfakeDataset
from src.data.augmentation import ImageAugmenter
from src.features.pipeline import HandcraftedFeatureExtractor
from src.models.classical import ClassicalBaselineModel
from src.models.custom_cnn import build_custom_cnn
from src.models.transfer_learning import build_transfer_learning_model
from src.models.fusion import build_fusion_model
from src.models.trainer import ModelTrainer
from src.evaluation.metrics import ModelEvaluator
from src.evaluation.visualize import Visualizer
from src.evaluation.gradcam import GradCAM

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pipeline_runner")

def run_pipeline():
    logger.info("Initializing Deepfake Detection Pipeline execution...")

    # 1. Prepare Dataset
    dataset_mgr = DeepfakeDataset(data_dir="data/raw", image_size=(256, 256), seed=42)
    raw_data_dir = dataset_mgr.generate_synthetic_fallback_dataset(samples_per_class=100)
    paths, labels = dataset_mgr.collect_image_paths(raw_data_dir, max_samples=200)

    splits = dataset_mgr.get_stratified_split(paths, labels, ratios=(0.70, 0.15, 0.15))
    train_paths, train_labels = splits["train"]
    val_paths, val_labels = splits["val"]
    test_paths, test_labels = splits["test"]

    # Load images in memory for feature extraction & TF dataset creation
    logger.info(f"Loading and preprocessing {len(paths)} total images...")
    X_train_imgs = np.array([dataset_mgr.load_and_preprocess_image(p) for p in train_paths])
    X_val_imgs = np.array([dataset_mgr.load_and_preprocess_image(p) for p in val_paths])
    X_test_imgs = np.array([dataset_mgr.load_and_preprocess_image(p) for p in test_paths])

    y_train = np.array(train_labels, dtype=int)
    y_val = np.array(val_labels, dtype=int)
    y_test = np.array(test_labels, dtype=int)

    # 2. Handcrafted Feature Extraction
    logger.info("Extracting handcrafted spectral, LBP, noise, and color feature vectors...")
    feature_pipeline = HandcraftedFeatureExtractor(n_pca_components=50)
    raw_train = feature_pipeline.extract_batch_raw(list(X_train_imgs))
    raw_val = feature_pipeline.extract_batch_raw(list(X_val_imgs))
    raw_test = feature_pipeline.extract_batch_raw(list(X_test_imgs))

    X_train_feats = feature_pipeline.fit_transform(raw_train)
    X_val_feats = feature_pipeline.transform(raw_val)
    X_test_feats = feature_pipeline.transform(raw_test)
    feature_pipeline.save(scaler_path="reports/scaler.pkl", pca_path="reports/pca.pkl")

    metrics_report = {}
    roc_data = {}
    visualizer = Visualizer(output_dir="reports/figures")

    # 3. Model 1: Classical Baseline (Random Forest)
    logger.info("Training Classical ML Baseline (Random Forest)...")
    classical_model = ClassicalBaselineModel(model_type="random_forest", n_estimators=100, max_depth=15, random_state=42)
    classical_model.train(X_train_feats, y_train)
    classical_metrics = classical_model.evaluate(X_test_feats, y_test)
    classical_model.save("reports/classical_model.pkl")
    metrics_report["Classical_Random_Forest"] = classical_metrics

    y_prob_classical = classical_model.predict_proba(X_test_feats)[:, 1]
    roc_data["Classical Random Forest"] = {"y_true": y_test, "y_prob": y_prob_classical}
    visualizer.plot_confusion_matrix(y_test, (y_prob_classical >= 0.5).astype(int), "Classical Random Forest")

    # 4. TensorFlow Datasets for Deep Learning
    train_ds = tf.data.Dataset.from_tensor_slices((X_train_imgs, y_train)).batch(32)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val_imgs, y_val)).batch(32)
    test_ds = tf.data.Dataset.from_tensor_slices((X_test_imgs, y_test)).batch(32)

    # 5. Model 2: Custom CNN from scratch
    logger.info("Training Custom CNN Model from Scratch...")
    custom_cnn = build_custom_cnn(input_shape=(256, 256, 3), learning_rate=0.001)
    cnn_trainer = ModelTrainer(custom_cnn, model_save_path="reports/custom_cnn.keras")
    cnn_trainer.train(train_ds, val_ds, epochs=1, patience=1)
    cnn_trainer.plot_training_curves("reports/figures/custom_cnn_training_curves.png")

    y_prob_cnn = custom_cnn.predict(X_test_imgs)[:, 0]
    cnn_metrics = ModelEvaluator.compute_metrics(y_test, y_prob_cnn)
    metrics_report["Custom_CNN"] = cnn_metrics
    roc_data["Custom CNN"] = {"y_true": y_test, "y_prob": y_prob_cnn}
    visualizer.plot_confusion_matrix(y_test, (y_prob_cnn >= 0.5).astype(int), "Custom CNN")

    # 6. Model 3: EfficientNetB0 Transfer Learning
    logger.info("Training Transfer Learning Model (EfficientNetB0)...")
    effnet = build_transfer_learning_model(backbone_name="EfficientNetB0", input_shape=(256, 256, 3), learning_rate=0.0005, fine_tune_at=100)
    eff_trainer = ModelTrainer(effnet, model_save_path="reports/efficientnet_model.keras")
    eff_trainer.train(train_ds, val_ds, epochs=1, patience=1)
    eff_trainer.plot_training_curves("reports/figures/efficientnet_training_curves.png")

    y_prob_eff = effnet.predict(X_test_imgs)[:, 0]
    eff_metrics = ModelEvaluator.compute_metrics(y_test, y_prob_eff)
    metrics_report["EfficientNetB0_Transfer"] = eff_metrics
    roc_data["EfficientNetB0 Transfer"] = {"y_true": y_test, "y_prob": y_prob_eff}
    visualizer.plot_confusion_matrix(y_test, (y_prob_eff >= 0.5).astype(int), "EfficientNetB0 Transfer")

    # 7. Model 4: Late Fusion Architecture
    logger.info("Training Late Fusion Model (CNN + Handcrafted Features)...")
    fusion_model = build_fusion_model(image_shape=(256, 256, 3), feature_dim=50, learning_rate=0.0003)
    
    train_ds_fusion = tf.data.Dataset.from_tensor_slices(({"image_input": X_train_imgs, "feature_input": X_train_feats}, y_train)).batch(32)
    val_ds_fusion = tf.data.Dataset.from_tensor_slices(({"image_input": X_val_imgs, "feature_input": X_val_feats}, y_val)).batch(32)
    
    fusion_trainer = ModelTrainer(fusion_model, model_save_path="reports/fusion_model.keras")
    fusion_trainer.train(train_ds_fusion, val_ds_fusion, epochs=1, patience=1)
    fusion_trainer.plot_training_curves("reports/figures/fusion_model_training_curves.png")

    y_prob_fusion = fusion_model.predict({"image_input": X_test_imgs, "feature_input": X_test_feats})[:, 0]
    fusion_metrics = ModelEvaluator.compute_metrics(y_test, y_prob_fusion)
    metrics_report["Late_Fusion_Model"] = fusion_metrics
    roc_data["Late Fusion Model"] = {"y_true": y_test, "y_prob": y_prob_fusion}
    visualizer.plot_confusion_matrix(y_test, (y_prob_fusion >= 0.5).astype(int), "Late Fusion Model")

    # 8. Render Comparative Plots
    visualizer.plot_roc_curves(roc_data)
    visualizer.plot_model_comparison_bar(metrics_report)

    # 9. Grad-CAM Visualization on Sample Test Image
    try:
        gradcam = GradCAM(custom_cnn)
        sample_img = X_test_imgs[0]
        heatmap = gradcam.compute_heatmap(np.expand_dims(sample_img, axis=0))
        heatmap_rgb, overlay_rgb = gradcam.overlay_heatmap(sample_img, heatmap)
        pred_score = float(y_prob_cnn[0])
        pred_label = "Fake" if pred_score >= 0.5 else "Real"
        visualizer.save_gradcam_panel(
            sample_img, heatmap_rgb, overlay_rgb, pred_label, pred_score, filename="gradcam_sample.png"
        )
    except Exception as e:
        logger.warning(f"Grad-CAM generation skipped or encountered exception: {e}")

    # 10. Save Metrics JSON Report
    ModelEvaluator.save_metrics_report(metrics_report, "reports/metrics.json")
    logger.info("Pipeline execution complete! True metrics and figures generated in reports/")

if __name__ == "__main__":
    run_pipeline()
