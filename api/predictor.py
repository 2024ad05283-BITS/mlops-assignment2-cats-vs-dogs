"""
predictor.py

Production-ready inference module.

Responsibilities
----------------
1. Load trained model.
2. Preprocess input image.
3. Run inference.
4. Return predicted label and probabilities.
"""

from pathlib import Path

import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms

from src.models.cnn import SimpleCNN
from src.dataset.constants import (
    IMAGE_SIZE,
    NUM_CLASSES,
)

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

MODEL_PATH = Path("artifacts/models/best_model.pt")

CLASS_NAMES = [
    "Cat",
    "Dog",
]

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# ------------------------------------------------------------
# Image Transform
# ------------------------------------------------------------

transform = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

# ------------------------------------------------------------
# Predictor
# ------------------------------------------------------------


class Predictor:

    def __init__(self):

        self.device = DEVICE

        self.model = self._load_model()

    # --------------------------------------------------------

    def _load_model(self):

        if not MODEL_PATH.exists():

            raise FileNotFoundError(
                f"Model not found:\n{MODEL_PATH}"
            )

        model = SimpleCNN(
            num_classes=NUM_CLASSES
        ).to(self.device)

        checkpoint = torch.load(
            MODEL_PATH,
            map_location=self.device,
        )

        # Supports trainer checkpoint
        if isinstance(checkpoint, dict):

            if "model_state_dict" in checkpoint:

                model.load_state_dict(
                    checkpoint["model_state_dict"]
                )

            else:

                model.load_state_dict(
                    checkpoint
                )

        else:

            model.load_state_dict(
                checkpoint
            )

        model.eval()

        return model

    # --------------------------------------------------------

    def preprocess(
        self,
        image: Image.Image,
    ):

        image = image.convert("RGB")

        tensor = transform(image)

        tensor = tensor.unsqueeze(0)

        return tensor.to(self.device)

    # --------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        image: Image.Image,
    ):

        input_tensor = self.preprocess(image)

        outputs = self.model(input_tensor)

        probabilities = F.softmax(
            outputs,
            dim=1,
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=1,
        )

        return {

            "label": CLASS_NAMES[
                prediction.item()
            ],

            "class_id": prediction.item(),

            "confidence": round(
                confidence.item(),
                4,
            ),

            "probabilities": {

                CLASS_NAMES[i]: round(
                    probabilities[0][i].item(),
                    4,
                )

                for i in range(NUM_CLASSES)

            },

        }


# ------------------------------------------------------------
# Smoke Test
# ------------------------------------------------------------

if __name__ == "__main__":

    predictor = Predictor()

    print("=" * 60)
    print("Predictor Loaded Successfully")
    print("=" * 60)

    print(f"Device : {predictor.device}")

    print(f"Model  : {MODEL_PATH}")