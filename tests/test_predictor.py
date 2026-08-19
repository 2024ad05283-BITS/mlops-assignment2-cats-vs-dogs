"""
test_predictor.py

Unit tests for Predictor.
"""

from pathlib import Path

from PIL import Image

from api.predictor import Predictor


def test_model_exists():
    """
    Verify trained model exists.
    """

    assert Path(
        "artifacts/models/best_model.pt"
    ).exists()


def test_predictor_load():
    """
    Verify predictor loads successfully.
    """

    predictor = Predictor()

    assert predictor.model is not None


def test_predict_output():
    """
    Verify prediction dictionary.
    """

    predictor = Predictor()

    image = Image.new(
        "RGB",
        (224, 224),
        color="white",
    )

    result = predictor.predict(image)

    assert isinstance(result, dict)

    assert "label" in result

    assert "confidence" in result

    assert "probabilities" in result


def test_probability_sum():
    """
    Verify probabilities sum to ~1.
    """

    predictor = Predictor()

    image = Image.new(
        "RGB",
        (224, 224),
    )

    result = predictor.predict(image)

    total = sum(
        result["probabilities"].values()
    )

    assert abs(total - 1.0) < 0.01


def test_prediction_label():
    """
    Verify label is valid.
    """

    predictor = Predictor()

    image = Image.new(
        "RGB",
        (224, 224),
    )

    result = predictor.predict(image)

    assert result["label"] in (
        "Cat",
        "Dog",
    )