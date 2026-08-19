"""
test_preprocessing.py

Unit tests for dataset preprocessing.
"""

from PIL import Image

from src.dataset.loader import get_transforms
from src.dataset.constants import (
    IMAGE_CHANNELS,
    IMAGE_SIZE,
)


def test_image_transform_shape():
    """
    Verify transformed image shape.
    """

    image = Image.new(
        "RGB",
        (300, 300),
        color="white",
    )

    transform = get_transforms(train=False)

    tensor = transform(image)

    assert tensor.shape == (
        IMAGE_CHANNELS,
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
    )


def test_image_tensor_type():
    """
    Verify output is tensor.
    """

    image = Image.new(
        "RGB",
        (224, 224),
        color="black",
    )

    transform = get_transforms(train=False)

    tensor = transform(image)

    assert tensor.ndim == 3


def test_image_channels():
    """
    Verify RGB channels.
    """

    image = Image.new(
        "RGB",
        (224, 224),
    )

    transform = get_transforms(False)

    tensor = transform(image)

    assert tensor.shape[0] == IMAGE_CHANNELS