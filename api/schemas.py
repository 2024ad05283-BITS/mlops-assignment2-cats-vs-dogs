"""
schemas.py

Pydantic request and response schemas for the
Cats vs Dogs inference API.
"""

from pydantic import BaseModel, Field


# ============================================================
# Health Response
# ============================================================

class HealthResponse(BaseModel):
    """
    Health check response.
    """

    status: str = Field(
        ...,
        example="healthy",
    )

    model_loaded: bool = Field(
        ...,
        example=True,
    )


# ============================================================
# Prediction Response
# ============================================================

class PredictionResponse(BaseModel):
    """
    Prediction response returned by the API.
    """

    label: str = Field(
        ...,
        example="Cat",
    )

    class_id: int = Field(
        ...,
        example=0,
    )

    confidence: float = Field(
        ...,
        example=0.9876,
    )

    probabilities: dict[str, float] = Field(
        ...,
        example={
            "Cat": 0.9876,
            "Dog": 0.0124,
        },
    )


# ============================================================
# Error Response
# ============================================================

class ErrorResponse(BaseModel):
    """
    Error response.
    """

    detail: str = Field(
        ...,
        example="Invalid image file.",
    )


# ============================================================
# Smoke Test
# ============================================================

if __name__ == "__main__":

    health = HealthResponse(
        status="healthy",
        model_loaded=True,
    )

    prediction = PredictionResponse(
        label="Dog",
        class_id=1,
        confidence=0.9987,
        probabilities={
            "Cat": 0.0013,
            "Dog": 0.9987,
        },
    )

    print("=" * 60)
    print("Health Schema")
    print("=" * 60)
    print(health.model_dump())

    print()

    print("=" * 60)
    print("Prediction Schema")
    print("=" * 60)
    print(prediction.model_dump())