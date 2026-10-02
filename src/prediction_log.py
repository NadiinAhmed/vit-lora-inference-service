"""Append-only JSON Lines logs, the raw material for monitoring.

predictions.jsonl - one line per prediction
feedback.jsonl    - one line per ground-truth label received later via POST /feedback

A prediction line records: when, which model, what it predicted, how confident, how
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
        self.feedback_path = directory / "feedback.jsonl"
        # /predict runs in a thread pool, so two requests can write at the same moment;
        # the lock keeps their lines from interleaving.
        self._lock = threading.Lock()
        # request_id -> predicted label, so feedback can be checked and scored instantly.
        # Rebuilt from the file at startup, so feedback still works after a restart.
        self._predicted_labels = {r["request_id"]: r["label"] for r in read_jsonl(self.predictions_path)}

    def log_prediction(self, record: dict[str, Any]) -> None:
        self._predicted_labels[record["request_id"]] = record["label"]
        self._append(self.predictions_path, record)

    def predicted_label(self, request_id: str) -> str | None:
        """The label predicted for request_id, or None if no such prediction was logged."""
        return self._predicted_labels.get(request_id)

    def log_feedback(self, request_id: str, true_label: str) -> None:
        self._append(self.feedback_path, {"request_id": request_id, "true_label": true_label})

    def _append(self, path: Path, record: dict[str, Any]) -> None:
        line = json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(), **record})
        try:
            with self._lock, path.open("a", encoding="utf-8") as file:
                file.write(line + "\n")
        except OSError:
            # Monitoring must never take the service down: the user still gets the prediction.
            logger.exception("Could not write monitoring record to %s", path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSON Lines file; a missing file means no records yet."""
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]
