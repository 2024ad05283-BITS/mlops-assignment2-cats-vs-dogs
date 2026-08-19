"""
plotter.py

Utility functions for plotting training curves.

Responsibilities
----------------
1. Plot training vs validation loss.
2. Plot training vs validation accuracy.
3. Save figures as PNG artifacts.

These plots can be logged to MLflow.
"""

from pathlib import Path

import matplotlib.pyplot as plt


def plot_loss_curve(
    history: dict,
    output_path: str = "artifacts/plots/loss_curve.png",
):
    """
    Plot training and validation loss.

    Parameters
    ----------
    history : dict
        Returned by Trainer.fit()

    output_path : str
        Output PNG path.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        history["train_loss"],
        label="Train Loss",
        linewidth=2,
    )

    plt.plot(
        history["val_loss"],
        label="Validation Loss",
        linewidth=2,
    )

    plt.title("Training vs Validation Loss")

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig(output_path)

    plt.close()

    print(f"[INFO] Loss curve saved to {output_path}")


def plot_accuracy_curve(
    history: dict,
    output_path: str = "artifacts/plots/accuracy_curve.png",
):
    """
    Plot training and validation accuracy.

    Parameters
    ----------
    history : dict
        Returned by Trainer.fit()

    output_path : str
        Output PNG path.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        history["train_accuracy"],
        label="Train Accuracy",
        linewidth=2,
    )

    plt.plot(
        history["val_accuracy"],
        label="Validation Accuracy",
        linewidth=2,
    )

    plt.title("Training vs Validation Accuracy")

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy (%)")

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig(output_path)

    plt.close()

    print(f"[INFO] Accuracy curve saved to {output_path}")


if __name__ == "__main__":

    dummy_history = {

        "train_loss": [1.2, 0.8, 0.5, 0.3],

        "val_loss": [1.3, 0.9, 0.6, 0.4],

        "train_accuracy": [52, 71, 86, 94],

        "val_accuracy": [50, 69, 84, 91],
    }

    plot_loss_curve(dummy_history)

    plot_accuracy_curve(dummy_history)

    print("Plotter smoke test completed.")