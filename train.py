
"""
train.py

Production-style training entry point.

Responsibilities
----------------
1. Build configuration
2. Prepare dataset
3. Build model
4. Train model
5. Evaluate model
6. Generate plots
7. Log experiments with MLflow
"""

import json
import logging
import sys
from pathlib import Path

import torch

# Dataset
from src.dataset.manager import DatasetManager

# Models
from src.models.cnn import SimpleCNN
from src.models.trainer import Trainer
from src.models.evaluator import Evaluator

# Experiment Tracking
from src.tracking.mlflow_logger import MLflowLogger

# Plot Utilities
from src.utils.plotter import (
    plot_loss_curve,
    plot_accuracy_curve,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def build_config() -> dict:
    """
    Central configuration for the training pipeline.
    """

    return {
        "batch_size": 32,
        "num_workers": 4, 
        "epochs": 10,
        "learning_rate": 1e-3,
    }


def main() -> int:
    """
    Main training pipeline.
    """

    print("=" * 60)
    print(">>> main() started <<<")
    print("=" * 60)

    try:

        print("[STEP 1] Building configuration...")
        config = build_config()
        print("✓ Configuration loaded.")

        print("[STEP 2] Creating DatasetManager...")
        dataset_manager = DatasetManager(config)
        print("✓ DatasetManager created.")

        print("[STEP 3] Preparing dataset...")
        train_loader, val_loader, test_loader = dataset_manager.prepare()
        print("✓ Dataset prepared.")

        print("[STEP 4] Creating CNN model...")
        model = SimpleCNN(num_classes=2)
        print("✓ Model created.")

        print("[STEP 5] Creating loss function...")
        criterion = torch.nn.CrossEntropyLoss()
        print("✓ Criterion created.")

        print("[STEP 6] Creating optimizer...")
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["learning_rate"],
        )
        print("✓ Optimizer created.")

        print("[STEP 7] Starting MLflow...")
        mlflow_logger = MLflowLogger(
            experiment_name="CatsVsDogs_M1"
        )

        mlflow_logger.start_run(
            run_name="Baseline_CNN"
        )

        mlflow_logger.log_params({
            "epochs": config["epochs"],
            "batch_size": config["batch_size"],
            "learning_rate": config["learning_rate"],
            "optimizer": "Adam",
            "loss_function": "CrossEntropyLoss",
            "model": "SimpleCNN",
        })

        print("✓ MLflow started.")

        print("[STEP 8] Creating Trainer...")
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            mlflow_logger=mlflow_logger,
        )
        print("✓ Trainer created.")

        print("[STEP 9] Training model...")
        history = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config["epochs"],
        )
        
        print("✓ Training completed.")

        print("[STEP 10] Plotting training curves...")
        plot_loss_curve(
            history,
            "artifacts/plots/loss_curve.png",
        )

        plot_accuracy_curve(
            history,
            "artifacts/plots/accuracy_curve.png",
        )

        print("✓ Plots generated.")

        print("[STEP 11] Evaluating model...")
        evaluator = Evaluator()

        metrics = evaluator.evaluate(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
        )

        print("✓ Evaluation completed.")

        print("[STEP 12] Saving confusion matrix...")
        evaluator.save_confusion_matrix(
            cm=metrics["confusion_matrix"],
            class_names=["Cat", "Dog"],
            output_path="artifacts/plots/confusion_matrix.png",
        )

        print("✓ Confusion matrix saved.")

        print("[STEP 13] Logging metrics to MLflow...")
        mlflow_logger.log_metrics({
            "test_loss": metrics["loss"],
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
        })

        mlflow_logger.log_artifact(
            "artifacts/plots/loss_curve.png"
        )

        mlflow_logger.log_artifact(
            "artifacts/plots/accuracy_curve.png"
        )

        mlflow_logger.log_artifact(
            "artifacts/plots/confusion_matrix.png"
        )

        mlflow_logger.log_model(
            model,
            artifact_path="cnn_model",
        )

        print("✓ MLflow logging completed.")

        mlflow_logger.end_run()

        print("=" * 60)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 60)

        return 0

    except Exception as e:

        print("=" * 60)
        print("❌ PIPELINE FAILED")
        print("=" * 60)

        print(type(e).__name__)
        print(e)

        import traceback
        traceback.print_exc()

        if "mlflow_logger" in locals():
            mlflow_logger.end_run()

        return 1
        
        
if __name__ == "__main__":
    main()