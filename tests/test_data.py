"""Dataset splitting, folder discovery and metrics (training utilities)."""

from collections import Counter
from pathlib import Path

import pytest
import torch
from PIL import Image

from training.data import find_class_root, scan_image_folder, stratified_split
from training.evaluation import compute_metrics


@pytest.fixture()
def image_folder(tmp_path: Path) -> Path:
    root = tmp_path / "wrapper" / "dataset"
    for name, count in {"glass": 40, "paper": 60, "trash": 20}.items():
        (root / name).mkdir(parents=True)
        for i in range(count):
            Image.new("RGB", (8, 8)).save(root / name / f"{i}.jpg")
    junk = tmp_path / "__MACOSX" / "dataset" / "glass"
    junk.mkdir(parents=True)
    (junk / "._0.jpg").write_bytes(b"x")
    return tmp_path


def test_find_class_root_skips_wrappers_and_macos_junk(image_folder: Path) -> None:
    assert find_class_root(image_folder) == image_folder / "wrapper" / "dataset"


def test_stratified_split_preserves_proportions_without_overlap(image_folder: Path) -> None:
    class_names, samples = scan_image_folder(find_class_root(image_folder))
    train, val, test = stratified_split(samples, 0.15, 0.15, seed=42)

    assert class_names == ["glass", "paper", "trash"]
    assert len(train) + len(val) + len(test) == len(samples)
    assert not ({p for p, _ in train} & {p for p, _ in val} or {p for p, _ in train} & {p for p, _ in test})
    assert Counter(label for _, label in test) == {0: 6, 1: 9, 2: 3}  # 15% of each class


def test_stratified_split_is_deterministic(image_folder: Path) -> None:
    _, samples = scan_image_folder(find_class_root(image_folder))
    assert stratified_split(samples, 0.15, 0.15, seed=7) == stratified_split(samples, 0.15, 0.15, seed=7)


def test_compute_metrics_known_case() -> None:
    labels = torch.tensor([0, 0, 1, 1])
    predictions = torch.tensor([0, 1, 1, 1])
    metrics = compute_metrics(predictions, labels, ["a", "b"])
    assert metrics["accuracy"] == 0.75
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]
    assert metrics["per_class_f1"] == {"a": 0.6667, "b": 0.8}
    assert metrics["macro_f1"] == 0.7333
