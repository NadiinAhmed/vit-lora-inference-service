"""FastAPI application exposing GET /health and POST /predict.

Request flow:  route -> validate upload -> decode image -> InferenceService -> response schema
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status

from src.config import get_settings
from src.inference import InferenceService
from src.preprocessing import InvalidImageError, decode_image
from src.schemas import ClassProbability, HealthResponse, PredictionResponse

logger = logging.getLogger("vit_service")


def create_app(service: InferenceService | None = None) -> FastAPI:
    """Build the app. Tests inject a small service; production loads the real model."""
    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Load once at startup. If loading fails the server refuses to start (fail fast)
        # rather than accepting traffic it cannot serve.
        start = time.perf_counter()
        app.state.service = service or InferenceService.from_settings(settings)
        logger.info("model_loaded seconds=%.2f classes=%s", time.perf_counter() - start, app.state.service.labels)
        yield

    app = FastAPI(
        title="ViT LoRA Inference Service",
        description="Waste-material image classification with a LoRA-fine-tuned, INT8-quantized Vision Transformer.",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        svc: InferenceService | None = getattr(request.app.state, "service", None)
        return HealthResponse(status="ok", model_loaded=svc is not None, classes=svc.labels if svc else [])

    # A plain `def` (not `async def`): model inference is CPU-bound and blocking, so
    # FastAPI runs it in a worker thread instead of freezing the event loop.
    @app.post(
        "/predict",
        response_model=PredictionResponse,
        responses={400: {"description": "Invalid image"}, 413: {"description": "File too large"},
                   415: {"description": "Not an image"}},
    )
    def predict(request: Request, file: UploadFile = File(..., description="Image file (JPEG, PNG, ...)")) -> PredictionResponse:
        if not (file.content_type or "").startswith("image/"):
            raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Upload must be an image.")

        data = file.file.read(settings.max_image_bytes + 1)  # read at most limit+1 bytes
        if len(data) > settings.max_image_bytes:
            raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE,
                                f"Image exceeds {settings.max_image_bytes} bytes.")
        try:
            image = decode_image(data)
        except InvalidImageError as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

        start = time.perf_counter()
        scores = request.app.state.service.predict(image)
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

        logger.info("prediction label=%s confidence=%.4f inference_ms=%.2f size=%dx%d",
                    scores[0].label, scores[0].confidence, elapsed_ms, *image.size)
        return PredictionResponse(
            label=scores[0].label,
            confidence=scores[0].confidence,
            top_k=[ClassProbability(label=s.label, confidence=s.confidence) for s in scores],
            inference_ms=elapsed_ms,
        )

    return app


app = create_app()
