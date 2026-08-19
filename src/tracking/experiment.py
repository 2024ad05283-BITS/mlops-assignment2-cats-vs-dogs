"""
MLflow experiment management.
"""
from typing import Optional
import mlflow

def set_tracking_uri(uri: str) -> None:
    mlflow.set_tracking_uri(uri)

def set_experiment(name: str) -> None:
    mlflow.set_experiment(name)

def get_experiment(name: str):
    return mlflow.get_experiment_by_name(name)

def create_experiment(name: str, artifact_location: Optional[str]=None):
    exp = mlflow.get_experiment_by_name(name)
    if exp:
        return exp.experiment_id
    return mlflow.create_experiment(name, artifact_location=artifact_location)
