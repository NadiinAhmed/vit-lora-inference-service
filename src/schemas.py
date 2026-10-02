"""Pydantic response schemas: they validate output and generate the OpenAPI docs."""

from pydantic import BaseModel, Field


class ClassProbability(BaseModel):
    label: str = Field(examples=["plastic"])
    confidence: float = Field(ge=0.0, le=1.0, examples=[0.9412])


class PredictionResponse(BaseModel):
    request_id: str = Field(description="Unique ID of this prediction, used to attach the true label later.",
                            examples=["3f2b9c0e8d4a4f6b9e1c2d3a4b5c6d7e"])
    label: str = Field(description="Most likely class.", examples=["plastic"])
    confidence: float = Field(ge=0.0, le=1.0, examples=[0.9412])
    top_k: list[ClassProbability] = Field(description="Top-k classes, highest first.")
    inference_ms: float = Field(description="Model latency for this request in milliseconds.")


class HealthResponse(BaseModel):
    status: str = Field(examples=["ok"])
    model_loaded: bool
    model_version: str = Field(examples=["model-v1.0.0"])
    classes: list[str]


class FeedbackRequest(BaseModel):
    request_id: str = Field(description="request_id returned by POST /predict.",
                            examples=["3f2b9c0e8d4a4f6b9e1c2d3a4b5c6d7e"])
    true_label: str = Field(description="The correct class, e.g. confirmed by a human.", examples=["glass"])


class FeedbackResponse(BaseModel):
    request_id: str
    predicted_label: str
    true_label: str
    correct: bool
