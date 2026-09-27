"""Pydantic response schemas: they validate output and generate the OpenAPI docs."""

from pydantic import BaseModel, Field


class ClassProbability(BaseModel):
    label: str = Field(examples=["plastic"])
    confidence: float = Field(ge=0.0, le=1.0, examples=[0.9412])


class PredictionResponse(BaseModel):
    label: str = Field(description="Most likely class.", examples=["plastic"])
    confidence: float = Field(ge=0.0, le=1.0, examples=[0.9412])
    top_k: list[ClassProbability] = Field(description="Top-k classes, highest first.")
    inference_ms: float = Field(description="Model latency for this request in milliseconds.")


class HealthResponse(BaseModel):
    status: str = Field(examples=["ok"])
    model_loaded: bool
    classes: list[str]
