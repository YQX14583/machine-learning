"""General utilities for reproducible CIFAR-10 experiments."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix

matplotlib.use("Agg")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    serializable = _make_json_ready(data)
    with Path(path).open("w", encoding="utf-8") as file_obj:
        json.dump(serializable, file_obj, ensure_ascii=False, indent=2)


def _make_json_ready(value: Any):
    if isinstance(value, dict):
        return {key: _make_json_ready(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_make_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_make_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def save_text(path: str | Path, content: str) -> None:
    with Path(path).open("w", encoding="utf-8") as file_obj:
        file_obj.write(content)


def save_results_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with Path(path).open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_training_curves(histories: dict[str, dict[str, list[float]]], output_path: str | Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for model_name, history in histories.items():
        axes[0].plot(history["train_loss"], label=f"{model_name} train")
        axes[0].plot(history["val_loss"], linestyle="--", label=f"{model_name} val")
        axes[1].plot(history["train_acc"], label=f"{model_name} train")
        axes[1].plot(history["val_acc"], linestyle="--", label=f"{model_name} val")

    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Accuracy Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix_figure(
    targets: np.ndarray,
    predictions: np.ndarray,
    class_names: list[str],
    output_path: str | Path,
    title: str,
) -> None:
    cm = confusion_matrix(targets, predictions)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(results: list[dict[str, Any]], output_path: str | Path) -> None:
    names = [row["model"] for row in results]
    accuracies = [row["test_accuracy"] for row in results]
    params_millions = [row["params"] / 1_000_000 for row in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(names, accuracies, color=["#5b8ff9", "#61dDAa", "#f6bd16"][: len(names)])
    axes[0].set_title("Test Accuracy")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(names, params_millions, color=["#5d7092", "#e86452", "#6dc8ec"][: len(names)])
    axes[1].set_title("Trainable Parameters")
    axes[1].set_ylabel("Millions")
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def compute_per_class_accuracy(
    targets: np.ndarray,
    predictions: np.ndarray,
    class_names: list[str],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for idx, class_name in enumerate(class_names):
        mask = targets == idx
        accuracy = float((predictions[mask] == idx).mean() * 100) if mask.any() else 0.0
        result[class_name] = round(accuracy, 2)
    return result


def plot_per_class_accuracy(
    per_model_scores: dict[str, dict[str, float]],
    output_path: str | Path,
) -> None:
    class_names = list(next(iter(per_model_scores.values())).keys())
    x = np.arange(len(class_names))
    width = 0.8 / len(per_model_scores)

    fig, ax = plt.subplots(figsize=(14, 6))
    for idx, (model_name, scores) in enumerate(per_model_scores.items()):
        values = [scores[class_name] for class_name in class_names]
        ax.bar(x + idx * width, values, width=width, label=model_name)

    ax.set_xticks(x + width * (len(per_model_scores) - 1) / 2)
    ax.set_xticklabels(class_names, rotation=30)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Per-Class Accuracy Comparison")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
