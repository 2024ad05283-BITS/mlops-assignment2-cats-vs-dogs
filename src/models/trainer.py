"""
trainer.py

Production-oriented trainer for M1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader


print("Trainer Imported Successfully")

class Trainer:
    """Encapsulates model training."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str | None = None,
        scheduler=None,
        mlflow_logger=None,
    ) -> None:
        """
        Initialize trainer.
        """

        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.mlflow_logger = mlflow_logger

    def train_one_epoch(
        self,
        loader: DataLoader,
        epoch: int = 1,
        total_epochs: int = 1,
    ) -> Dict[str, float]:
        """
        Train the model for one epoch.

        Prints training progress every 50 batches.
        """

        self.model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        total_batches = len(loader)

        for batch_idx, (images, labels) in enumerate(loader, start=1):

            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)

            loss = self.criterion(outputs, labels)

            loss.backward()

            self.optimizer.step()

            batch_size = images.size(0)

            running_loss += loss.item() * batch_size

            _, predicted = outputs.max(1)

            total += batch_size

            correct += predicted.eq(labels).sum().item()

            # =====================================================
            # Training Progress
            # =====================================================

            if batch_idx % 50 == 0 or batch_idx == total_batches:

                current_loss = running_loss / max(total, 1)

                current_accuracy = (
                    100.0 * correct / max(total, 1)
                )

                print(
                    f"Epoch {epoch:03d}/{total_epochs} | "
                    f"Batch {batch_idx:04d}/{total_batches} | "
                    f"Loss: {current_loss:.4f} | "
                    f"Acc: {current_accuracy:.2f}%",
                    flush=True,
                )

        epoch_loss = running_loss / max(total, 1)

        epoch_accuracy = (
            100.0 * correct / max(total, 1)
        )

        return {
            "loss": epoch_loss,
            "accuracy": epoch_accuracy,
        }
        
    @torch.no_grad()
    def validate(
        self,
        loader: DataLoader,
    ) -> Dict[str, float]:
        """
        Validate the model.
        """

        self.model.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in loader:

            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)

            loss = self.criterion(outputs, labels)

            batch_size = images.size(0)

            running_loss += loss.item() * batch_size

            _, predicted = outputs.max(1)

            total += batch_size

            correct += predicted.eq(labels).sum().item()
            
        epoch_loss = running_loss / max(total, 1)

        epoch_accuracy = 100.0 * correct / max(total, 1)

        return {
            "loss": epoch_loss,
            "accuracy": epoch_accuracy,
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        checkpoint_dir: str | Path = "artifacts/models",
        patience: int = 5,
    ):
        """
        Train the model.
        """
        print(">>> Entered fit() <<<")
        checkpoint_dir = Path(checkpoint_dir)

        checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }

        best_acc = 0.0
        patience_counter = 0

        for epoch in range(1, epochs + 1):

            #train_metrics = self.train_one_epoch(train_loader)
            train_metrics = self.train_one_epoch(
                train_loader,
                epoch=epoch,
                total_epochs=epochs,
            )
            val_metrics = self.validate(val_loader)

            history["train_loss"].append(train_metrics["loss"])
            history["train_accuracy"].append(train_metrics["accuracy"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_accuracy"].append(val_metrics["accuracy"])

            print(
                f"Epoch {epoch:03d}/{epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Train Acc: {train_metrics['accuracy']:.2f}% | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.2f}%"
            )

            if self.mlflow_logger is not None:

                self.mlflow_logger.log_metrics(
                    {
                        "train_loss": train_metrics["loss"],
                        "train_accuracy": train_metrics["accuracy"],
                        "val_loss": val_metrics["loss"],
                        "val_accuracy": val_metrics["accuracy"],
                    },
                    step=epoch,
                )

            if self.scheduler is not None:
                self.scheduler.step()

            if val_metrics["accuracy"] > best_acc:

                best_acc = val_metrics["accuracy"]

                patience_counter = 0

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "best_accuracy": best_acc,
                    },
                    checkpoint_dir / "best_model.pt",
                )

                print(
                    f"Best model saved "
                    f"(Val Accuracy = {best_acc:.2f}%)"
                )

            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_accuracy": best_acc,
            },
            checkpoint_dir / "last_model.pt",
        )

        print("=" * 60)
        print("Training Completed")
        print(f"Best Validation Accuracy : {best_acc:.2f}%")
        print(f"Best Model : {checkpoint_dir / 'best_model.pt'}")
        print(f"Last Model : {checkpoint_dir / 'last_model.pt'}")
        print("=" * 60)

        return history

if __name__ == "__main__":

    from torchvision.datasets import FakeData
    from torchvision.transforms import ToTensor
    from torch.utils.data import DataLoader

    from src.models.cnn import SimpleCNN

    model = SimpleCNN()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    criterion = torch.nn.CrossEntropyLoss()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
    )

    dataset = FakeData(
        size=64,
        image_size=(3, 224, 224),
        num_classes=2,
        transform=ToTensor(),
    )

    loader = DataLoader(
        dataset,
        batch_size=8,
    )

    history = trainer.fit(
        train_loader=loader,
        val_loader=loader,
        epochs=2,
    )

    print(history)