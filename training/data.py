"""TrashNet download, deterministic stratified splits, label mapping and Dataset.

TrashNet ships as a folder-per-class zip without predefined splits, so we create
train/validation/test ourselves: stratified (each split keeps the class
proportions) and seeded (the same split every run, which quantize.py relies on
to evaluate on exactly the same test images).
"""

from __future__ import annotations

import random
import zipfile
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from torch.utils.data import Dataset

from src.preprocessing import ImagePreprocessor, open_image

TRASHNET_REPO = "garythung/trashnet"
TRASHNET_ZIP = "dataset-resized.zip"  # ~43 MB; the repo's other zip is 3.6 GB originals
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

Sample = tuple[Path, int]


@dataclass(frozen=True)
class DatasetSplits:
    class_names: list[str]
    train: list[Sample]
    validation: list[Sample]
    test: list[Sample]


def download_trashnet(data_dir: Path) -> Path:
    """Download and extract TrashNet once; return the folder containing class sub-folders."""
    extract_dir = data_dir / "trashnet"
    if not extract_dir.exists():
        zip_path = hf_hub_download(TRASHNET_REPO, TRASHNET_ZIP, repo_type="dataset")
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(extract_dir)
    return find_class_root(extract_dir)


def find_class_root(directory: Path) -> Path:
    """Locate the directory whose sub-folders are the classes (zips often add wrapper folders)."""
    candidates = [directory, *sorted(p for p in directory.rglob("*") if p.is_dir())]
    for candidate in candidates:
        if "__MACOSX" in candidate.parts:
            continue
        class_dirs = [d for d in candidate.iterdir() if d.is_dir() and not d.name.startswith((".", "__"))]
        if len(class_dirs) >= 2 and all(_has_images(d) for d in class_dirs):
            return candidate
    raise FileNotFoundError(f"No class folders with images found under '{directory}'.")


def _has_images(directory: Path) -> bool:
    return any(p.suffix.lower() in IMAGE_SUFFIXES for p in directory.iterdir())


def scan_image_folder(root: Path) -> tuple[list[str], list[Sample]]:
    """Return sorted class names and (image_path, label_id) samples."""
    class_names = sorted(d.name for d in root.iterdir() if d.is_dir() and not d.name.startswith((".", "__")))
    samples = [
        (path, label_id)
        for label_id, name in enumerate(class_names)
        for path in sorted((root / name).iterdir())
        if path.suffix.lower() in IMAGE_SUFFIXES
    ]
    return class_names, samples


def stratified_split(samples: list[Sample], val_fraction: float, test_fraction: float, seed: int
                     ) -> tuple[list[Sample], list[Sample], list[Sample]]:
    """Split each class separately so every split keeps the original class balance."""
    rng = random.Random(seed)
    by_class: dict[int, list[Sample]] = {}
    for sample in samples:
        by_class.setdefault(sample[1], []).append(sample)

    train, validation, test = [], [], []
    for label in sorted(by_class):
        items = by_class[label][:]
        rng.shuffle(items)
        n_test = round(len(items) * test_fraction)
        n_val = round(len(items) * val_fraction)
        test += items[:n_test]
        validation += items[n_test:n_test + n_val]
        train += items[n_test + n_val:]
    return train, validation, test


def load_splits(data_root: Path, val_fraction: float = 0.15, test_fraction: float = 0.15, seed: int = 42
                ) -> DatasetSplits:
    class_names, samples = scan_image_folder(data_root)
    train, validation, test = stratified_split(samples, val_fraction, test_fraction, seed)
    return DatasetSplits(class_names, train, validation, test)


class ImageClassificationDataset(Dataset):
    """Loads images lazily and applies the shared preprocessing.

    Training uses a random horizontal flip: a mirrored bottle is still a bottle, so
    this adds variety without changing the label. Evaluation never augments.
    """

    def __init__(self, samples: list[Sample], preprocessor: ImagePreprocessor, augment: bool = False) -> None:
        self._samples = samples
        self._preprocessor = preprocessor
        self._augment = augment

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        path, label = self._samples[index]
        image = open_image(path)
        pixel_values = self._preprocessor([image])[0]
        if self._augment and torch.rand(1).item() < 0.5:
            pixel_values = pixel_values.flip(-1)  # horizontal flip
        return {"pixel_values": pixel_values, "labels": torch.tensor(label)}
