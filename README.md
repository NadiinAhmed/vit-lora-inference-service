# ViT LoRA Inference Service

An end-to-end Vision Transformer image classification service featuring parameter-efficient LoRA fine-tuning, model quantization, FastAPI inference, Docker containerization, and automated testing.

> 🚧 Work in progress — full documentation, results and usage are added as each phase is completed.

## Pipeline

**Training (offline):** Dataset → Preprocessing → Pretrained ViT → LoRA fine-tuning → Best LoRA adapter → Merge → INT8 quantization

**Serving (Docker):** Image → Preprocessing → Quantized ViT → Prediction → FastAPI

## Project Structure

```
src/            Serving code + shared components (config, preprocessing, model loading)
training/       Offline pipeline: data, LoRA training, quantization (never shipped in Docker)
tests/          pytest suite
models/         Generated artifacts (git-ignored)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-train.txt -r requirements-dev.txt
cp .env.example .env
pytest
```
