"""Append-only JSON Lines log of every prediction, the raw material for monitoring.

One line per request: when, which model, what it predicted, how confident, how
long it took, and a few statistics of the input image. The image itself is NOT
stored: monitoring needs distributions, not users' photos (privacy-first logging).

JSON Lines (one JSON object per line) is easy to append to, to read with pandas
or a few lines of Python, and to inspect with any text editor.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("vit_service")


class PredictionLog:
    def __init__(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.predictions_path = directory / "predictions.jsonl"
        # /predict runs in a thread pool, so two requests can write at the same moment;
        # the lock keeps their lines from interleaving.
        self._lock = threading.Lock()

    def log_prediction(self, record: dict[str, Any]) -> None:
        self._append(self.predictions_path, record)

    def _append(self, path: Path, record: dict[str, Any]) -> None:
        line = json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(), **record})
        try:
            with self._lock, path.open("a", encoding="utf-8") as file:
                file.write(line + "\n")
        except OSError:
            # Monitoring must never take the service down: the user still gets the prediction.
            logger.exception("Could not write monitoring record to %s", path)
