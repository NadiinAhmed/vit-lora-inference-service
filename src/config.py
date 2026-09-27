"""Central, typed configuration shared by training, quantization and serving.

Values come from (highest priority first): environment variables prefixed with
``APP_``, a local ``.env`` file, then the defaults below. This keeps paths and
limits out of the code so the same image runs locally and in Docker.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings loaded once and reused everywhere."""

    model_config = SettingsConfigDict(env_prefix="APP_", env_file=".env", extra="ignore")

    # --- Model -------------------------------------------------------------
    base_model_name: str = "google/vit-base-patch16-224-in21k"
    model_dir: Path = PROJECT_ROOT / "models"

    # --- Serving -----------------------------------------------------------
    top_k: int = Field(default=3, ge=1)
    max_image_bytes: int = Field(default=5 * 1024 * 1024, gt=0)  # 5 MB upload limit
    log_level: str = "INFO"

    # --- Derived artifact locations ---------------------------------------
    @property
    def lora_adapter_dir(self) -> Path:
        """Best LoRA adapter produced by training (small, adapter weights only)."""
        return self.model_dir / "lora_best"

    @property
    def quantized_model_dir(self) -> Path:
        """Serving artifact: model config, preprocessor config and INT8 weights."""
        return self.model_dir / "quantized"

    @property
    def quantized_weights_path(self) -> Path:
        return self.quantized_model_dir / "model_int8.pt"


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (read the environment only once)."""
    return Settings()
