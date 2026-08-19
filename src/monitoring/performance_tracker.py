"""
performance_tracker.py

Production-style post-deployment performance tracker.

Responsibilities
----------------
1. Store predictions and true labels.
2. Compute Accuracy, Precision, Recall, F1.
3. Save evaluation report.
4. Save predictions CSV.
"""

from pathlib import Path

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


class PerformanceTracker:
    """
    Track deployed model performance.
    """

    def __init__(
        self,
        output_dir: str = "artifacts/reports",
    ):

        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.y_true = []
        self.y_pred = []

    # ---------------------------------------------------------

    def add_prediction(
        self,
        true_label,
        predicted_label,
    ):
        """
        Add one prediction.
        """

        self.y_true.append(true_label)

        self.y_pred.append(predicted_label)

    # ---------------------------------------------------------

    def compute_metrics(self):
        """
        Compute evaluation metrics.
        """

        if len(self.y_true) == 0:

            raise ValueError(
                "No predictions available."
            )

        metrics = {

            "accuracy":
                accuracy_score(
                    self.y_true,
                    self.y_pred,
                ),

            "precision":
                precision_score(
                    self.y_true,
                    self.y_pred,
                    average="weighted",
                    zero_division=0,
                ),

            "recall":
                recall_score(
                    self.y_true,
                    self.y_pred,
                    average="weighted",
                    zero_division=0,
                ),

            "f1_score":
                f1_score(
                    self.y_true,
                    self.y_pred,
                    average="weighted",
                    zero_division=0,
                ),
        }

        metrics["classification_report"] = (
            classification_report(
                self.y_true,
                self.y_pred,
                zero_division=0,
            )
        )

        metrics["confusion_matrix"] = (
            confusion_matrix(
                self.y_true,
                self.y_pred,
            ).tolist()
        )

        return metrics

    # ---------------------------------------------------------

    def save_predictions(self):
        """
        Save predictions CSV.
        """

        df = pd.DataFrame(

            {

                "true_label": self.y_true,

                "predicted_label": self.y_pred,

            }

        )

        csv_path = (
            self.output_dir /
            "predictions.csv"
        )

        df.to_csv(
            csv_path,
            index=False,
        )

        return csv_path

    # ---------------------------------------------------------

    def save_report(
        self,
        metrics,
    ):
        """
        Save evaluation report.
        """

        report_path = (
            self.output_dir /
            "evaluation_report.txt"
        )

        with open(
            report_path,
            "w",
        ) as f:

            f.write("=" * 60 + "\n")

            f.write(
                "POST DEPLOYMENT MODEL REPORT\n"
            )

            f.write("=" * 60 + "\n\n")

            f.write(
                f"Accuracy : {metrics['accuracy']:.4f}\n"
            )

            f.write(
                f"Precision : {metrics['precision']:.4f}\n"
            )

            f.write(
                f"Recall : {metrics['recall']:.4f}\n"
            )

            f.write(
                f"F1 Score : {metrics['f1_score']:.4f}\n\n"
            )

            f.write(
                metrics["classification_report"]
            )

        return report_path

    # ---------------------------------------------------------

    def evaluate(self):
        """
        Complete evaluation pipeline.
        """

        metrics = self.compute_metrics()

        csv_path = self.save_predictions()

        report_path = self.save_report(
            metrics,
        )

        print("=" * 60)
        print("Performance Tracking Completed")
        print("=" * 60)

        print(
            f"Accuracy : {metrics['accuracy']:.4f}"
        )

        print(
            f"Precision : {metrics['precision']:.4f}"
        )

        print(
            f"Recall : {metrics['recall']:.4f}"
        )

        print(
            f"F1 Score : {metrics['f1_score']:.4f}"
        )

        print()

        print(
            f"Predictions : {csv_path}"
        )

        print(
            f"Report : {report_path}"
        )

        print("=" * 60)

        return metrics


# ==========================================================
# Smoke Test
# ==========================================================

if __name__ == "__main__":

    tracker = PerformanceTracker()

    tracker.add_prediction("Cat", "Cat")
    tracker.add_prediction("Dog", "Dog")
    tracker.add_prediction("Cat", "Dog")
    tracker.add_prediction("Dog", "Dog")
    tracker.add_prediction("Cat", "Cat")

    tracker.evaluate()
    