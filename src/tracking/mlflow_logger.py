"""
mlflow_logger.py

Production-ready MLflow logging wrapper for M1.
"""

from pathlib import Path
from typing import Any, Dict

import mlflow
import mlflow.pytorch


class MLflowLogger:
    def __init__(self, experiment_name: str = "cats_vs_dogs_m1"):
        mlflow.set_experiment(experiment_name)

    def start_run(self, run_name: str | None = None):
        mlflow.start_run(run_name=run_name)

    def end_run(self):
        if mlflow.active_run():
            mlflow.end_run()

    def log_params(self, params: Dict[str, Any]):
        mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float], step: int | None = None):
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v), step=step)

    def log_artifact(self, artifact_path: str):
        p = Path(artifact_path)
        if p.exists():
            mlflow.log_artifact(str(p))

    def log_artifacts(self, artifact_dir: str):
        p = Path(artifact_dir)
        if p.exists():
            mlflow.log_artifacts(str(p))

    def log_model(self, model, artifact_path: str = "model"):
        mlflow.pytorch.log_model(model, artifact_path)

    def set_tags(self, tags: Dict[str, str]):
        mlflow.set_tags(tags)

    def log_training_summary(
        self,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        model=None,
        model_name: str = "model",
        artifacts_dir: str | None = None,
    ):
        self.log_params(params)
        self.log_metrics(metrics)
        if model is not None:
            self.log_model(model, model_name)
        if artifacts_dir:
            self.log_artifacts(artifacts_dir)


if __name__ == "__main__":
    logger = MLflowLogger("mlops_assignment_m1")
    logger.start_run("smoke_test")
    logger.log_params({"epochs": 10, "batch_size": 32})
    logger.log_metrics({"accuracy": 0.95, "loss": 0.12}, step=1)
    logger.set_tags({"stage": "M1"})
    logger.end_run()
    print("MLflow logger smoke test completed.")
