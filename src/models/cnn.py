"""
cnn.py

Baseline CNN model for Cats vs Dogs Classification.

Features
--------
- Simple CNN architecture
- Kaiming/Xavier weight initialization
- Parameter counting
- Model information
- Device-aware smoke test
"""

import torch
import torch.nn as nn

from src.dataset.constants import (
    IMAGE_CHANNELS,
    IMAGE_SIZE,
    NUM_CLASSES,
)


class SimpleCNN(nn.Module):
    """
    Baseline CNN model.
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        # ============================================================
        # Feature Extractor
        # ============================================================

        self.features = nn.Sequential(

            nn.Conv2d(
                IMAGE_CHANNELS,
                32,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # ============================================================
        # Classifier
        # ============================================================

        self.classifier = nn.Sequential(
        
            #nn.AdaptiveAvgPool2d((1, 1)),
            
            nn.Flatten(),

            nn.Linear(
                128 * 28 * 28,
                256,
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(0.5),

            nn.Linear(
                256,
                num_classes,
            ),
        )

        # Initialize weights

        self._initialize_weights()

    # ================================================================
    # Forward
    # ================================================================

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x

    # ================================================================
    # Weight Initialization
    # ================================================================

    def _initialize_weights(self):
        """
        Initialize model weights.
        """

        for module in self.modules():

            if isinstance(module, nn.Conv2d):

                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Linear):

                nn.init.xavier_uniform_(module.weight)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # ================================================================
    # Utilities
    # ================================================================

    def count_parameters(self) -> int:
        """
        Returns the number of trainable parameters.
        """

        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def model_info(self):
        """
        Print model information.
        """

        print("=" * 60)
        print("MODEL INFORMATION")
        print("=" * 60)

        print(f"Model Name        : {self.__class__.__name__}")
        print(f"Input Channels    : {IMAGE_CHANNELS}")
        print(f"Input Resolution  : {IMAGE_SIZE[0]} x {IMAGE_SIZE[1]}")
        print(f"Number of Classes : {NUM_CLASSES}")
        print(f"Trainable Params  : {self.count_parameters():,}")

        print("=" * 60)


# ===================================================================
# Smoke Test
# ===================================================================

if __name__ == "__main__":

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = SimpleCNN().to(device)

    model.model_info()

    dummy_input = torch.randn(
        1,
        IMAGE_CHANNELS,
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        device=device,
    )

    output = model(dummy_input)

    print(f"Input Shape  : {dummy_input.shape}")
    print(f"Output Shape : {output.shape}")
    print(f"Device       : {device}")