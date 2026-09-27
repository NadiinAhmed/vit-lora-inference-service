"""Configuration tests: defaults are sane and environment variables override them."""

from pathlib import Path

import pytest

from src.config import Settings


def test_defaults() -> None:
    settings = Settings(_env_file=None)
    assert settings.base_model_name == "google/vit-base-patch16-224-in21k"
    assert settings.top_k == 3
    assert settings.quantized_weights_path == settings.model_dir / "quantized" / "model_int8.pt"


def test_environment_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("APP_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("APP_TOP_K", "5")
    settings = Settings(_env_file=None)
    assert settings.model_dir == tmp_path
    assert settings.top_k == 5
    assert settings.lora_adapter_dir == tmp_path / "lora_best"


def test_invalid_value_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_TOP_K", "0")
    with pytest.raises(ValueError):
        Settings(_env_file=None)
