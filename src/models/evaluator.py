"""
evaluator.py

Production-style evaluator for binary image classification.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    ConfusionMatrixDisplay,
)


class Evaluator:
    """Evaluate a trained PyTorch model."""

    def __init__(self, device=None):

        self.device = torch.device(
            device or
            ("cuda" if torch.cuda.is_available() else "cpu")
        )

    @torch.no_grad()
    def evaluate(self, model, dataloader, criterion=None) -> Dict[str, float]:
        model.eval()
        model.to(self.device)

        y_true = []
        y_pred = []

        running_loss = 0.0

        for images, labels in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = model(images)

            if criterion is not None:
                running_loss += (
                    criterion(outputs, labels).item()
                    * images.size(0)
                )

            preds = torch.argmax(outputs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

        metrics = {
            "loss": running_loss / len(dataloader.dataset),
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="binary", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="binary", zero_division=0),
            "f1_score": f1_score(y_true, y_pred, average="binary", zero_division=0),
        }

        metrics["classification_report"] = classification_report(
            y_true, y_pred, digits=4, zero_division=0
        )

        cm = confusion_matrix(y_true, y_pred)

        metrics["confusion_matrix"] = cm
        return metrics

    def save_confusion_matrix(self, cm, class_names, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=class_names)
        fig, ax = plt.subplots(
            figsize=(6,6)
        )

        disp.plot(
            ax=ax,
            values_format="d",
        )
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close(fig)
