"""Utilities for loading and preprocessing the CIFAR-10 dataset."""

from __future__ import annotations

import os
import pickle
import tarfile
import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


@dataclass(slots=True)
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    class_names: list[str]
    train_size: int
    val_size: int
    test_size: int


class CIFAR10ArrayDataset(Dataset):
    """Dataset wrapper backed by numpy arrays."""

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        indices: Iterable[int] | None = None,
        transform=None,
    ) -> None:
        self.images = images
        self.labels = labels
        self.indices = np.array(list(indices)) if indices is not None else np.arange(len(images))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        real_idx = int(self.indices[idx])
        image = self.images[real_idx]
        label = int(self.labels[real_idx])

        if self.transform is not None:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        return image, label


class CIFAR10DataLoader:
    """Download, parse, split, and package CIFAR-10 data loaders."""

    def __init__(self, data_dir: str | os.PathLike[str]) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.batch_dir = self._prepare_dataset()

    def _prepare_dataset(self) -> Path:
        nested_dir = self.data_dir / "cifar-10-batches-py"
        flat_has_batches = (self.data_dir / "data_batch_1").exists() and (self.data_dir / "test_batch").exists()

        if nested_dir.exists():
            return nested_dir
        if flat_has_batches:
            return self.data_dir

        archive_path = self.data_dir / "cifar-10-python.tar.gz"
        print("Downloading CIFAR-10 from the official Toronto website...")
        urllib.request.urlretrieve(CIFAR10_URL, archive_path)

        print("Extracting dataset...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(self.data_dir)

        archive_path.unlink(missing_ok=True)
        return nested_dir

    def _load_batch(self, file_path: Path) -> tuple[np.ndarray, np.ndarray]:
        with file_path.open("rb") as file_obj:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="dtype\\(\\): align should be passed.*")
                batch = pickle.load(file_obj, encoding="latin1")
        data_key = "data" if "data" in batch else b"data"
        labels_key = "labels" if "labels" in batch else b"labels"
        images = batch[data_key].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        labels = np.array(batch[labels_key], dtype=np.int64)
        return images, labels

    def load_train_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        images, labels = [], []
        for batch_idx in range(1, 6):
            batch_images, batch_labels = self._load_batch(self.batch_dir / f"data_batch_{batch_idx}")
            images.append(batch_images)
            labels.append(batch_labels)
        return np.concatenate(images, axis=0), np.concatenate(labels, axis=0)

    def load_test_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return self._load_batch(self.batch_dir / "test_batch")

    def get_class_names(self) -> list[str]:
        meta_path = self.batch_dir / "batches.meta"
        with meta_path.open("rb") as file_obj:
            meta = pickle.load(file_obj, encoding="latin1")
        label_key = "label_names" if "label_names" in meta else b"label_names"
        names = meta[label_key]
        return [name.decode("utf-8") if isinstance(name, bytes) else str(name) for name in names]

    @staticmethod
    def build_transforms(use_autoaugment: bool = False):
        train_ops = [
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        if use_autoaugment:
            train_ops.append(transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10))
        train_ops.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            ]
        )

        eval_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            ]
        )

        return transforms.Compose(train_ops), eval_transform

    def create_data_loaders(
        self,
        batch_size: int = 128,
        val_ratio: float = 0.1,
        num_workers: int = 0,
        seed: int = 42,
        use_autoaugment: bool = False,
        subset_size: int | None = None,
    ) -> DataBundle:
        train_images, train_labels = self.load_train_arrays()
        test_images, test_labels = self.load_test_arrays()
        class_names = self.get_class_names() if (self.batch_dir / "batches.meta").exists() else CLASS_NAMES

        if subset_size is not None:
            subset_size = max(10, min(subset_size, len(train_images)))
            rng = np.random.default_rng(seed)
            subset_indices = np.sort(rng.choice(len(train_images), size=subset_size, replace=False))
            train_images = train_images[subset_indices]
            train_labels = train_labels[subset_indices]

        train_transform, eval_transform = self.build_transforms(use_autoaugment=use_autoaugment)
        rng = np.random.default_rng(seed)

        train_indices: list[int] = []
        val_indices: list[int] = []
        for class_idx in range(len(class_names)):
            class_members = np.where(train_labels == class_idx)[0]
            rng.shuffle(class_members)
            split_idx = max(1, int(len(class_members) * (1 - val_ratio)))
            train_indices.extend(class_members[:split_idx].tolist())
            val_indices.extend(class_members[split_idx:].tolist())

        rng.shuffle(train_indices)
        rng.shuffle(val_indices)

        train_dataset = CIFAR10ArrayDataset(train_images, train_labels, train_indices, train_transform)
        val_dataset = CIFAR10ArrayDataset(train_images, train_labels, val_indices, eval_transform)
        test_dataset = CIFAR10ArrayDataset(test_images, test_labels, transform=eval_transform)

        generator = torch.Generator()
        generator.manual_seed(seed)
        pin_memory = torch.cuda.is_available()

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            generator=generator,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        return DataBundle(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            class_names=class_names,
            train_size=len(train_dataset),
            val_size=len(val_dataset),
            test_size=len(test_dataset),
        )
