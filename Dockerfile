# Inference-only image: FastAPI + INT8-quantized ViT, CPU.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_OFFLINE=1

WORKDIR /app

# 1) CPU-only PyTorch (~200 MB) instead of the default CUDA build (several GB).
#    Its own layer: it changes rarely, so Docker's cache skips this slow step on rebuilds.
RUN pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu

# 2) Remaining serving dependencies (torch==2.14.0 is already satisfied by 2.14.0+cpu).
COPY requirements.txt .
RUN pip install -r requirements.txt

# 3) Application code and the quantized model (copied last: they change most often).
RUN useradd --create-home --uid 1000 appuser
COPY --chown=1000:1000 src/ ./src/
COPY --chown=1000:1000 models/quantized/ ./models/quantized/
# Monitoring logs (predictions/feedback). Mount a volume here to keep them after the container stops.
RUN mkdir /app/logs && chown 1000:1000 /app/logs
USER 1000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"]

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
