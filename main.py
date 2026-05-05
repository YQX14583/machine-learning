"""Entry point for training the single CNN model on CIFAR-10."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import torch

from data_loader import CIFAR10DataLoader
from models import get_model
from trainer import ModelTrainer
from utils import (
    compute_per_class_accuracy,
    count_parameters,
    ensure_dir,
    plot_confusion_matrix_figure,
    plot_training_curves,
    save_json,
    save_results_csv,
    save_text,
    set_seed,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a single optimized CNN on CIFAR-10.")
    parser.add_argument("--data-dir", default="data", help="Path to the CIFAR-10 data directory.")
    parser.add_argument("--output-root", default="outputs", help="Directory used to store experiment results.")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--subset-size",
        type=int,
        default=None,
        help="Optional small subset for smoke tests or quick demonstrations.",
    )
    return parser


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_training_config(epochs: int) -> dict:
    return {
        "epochs": epochs,
        "optimizer": "adamw",
        "scheduler": "cosine",
        "learning_rate": 7e-4,
        "weight_decay": 8e-4,
        "label_smoothing": 0.1,
        "early_stopping_patience": 10,
        "grad_clip": 1.0,
        "use_autoaugment": True,
    }


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)

    run_dir = ensure_dir(Path(args.output_root) / datetime.now().strftime("%Y%m%d_%H%M%S"))
    checkpoint_dir = ensure_dir(run_dir / "checkpoints")
    figure_dir = ensure_dir(run_dir / "figures")
    report_dir = ensure_dir(run_dir / "reports")

    device = get_device()
    config = get_training_config(args.epochs)
    print(f"Using device: {device}")
    print(f"Saving results to: {run_dir.resolve()}")

    loader_helper = CIFAR10DataLoader(args.data_dir)
    data_bundle = loader_helper.create_data_loaders(
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        num_workers=args.num_workers,
        seed=args.seed,
        use_autoaugment=True,
        subset_size=args.subset_size,
    )

    print(
        f"Dataset split -> train: {data_bundle.train_size}, "
        f"val: {data_bundle.val_size}, test: {data_bundle.test_size}"
    )

    model = get_model(num_classes=len(data_bundle.class_names))
    trainer = ModelTrainer(
        model=model,
        model_name="cnn",
        device=device,
        output_dir=checkpoint_dir,
        config=config,
    )

    train_result = trainer.train(data_bundle.train_loader, data_bundle.val_loader)
    test_metrics = trainer.evaluate(data_bundle.test_loader, data_bundle.class_names)
    per_class_accuracy = compute_per_class_accuracy(
        test_metrics["targets"], test_metrics["predictions"], data_bundle.class_names
    )

    plot_training_curves({"cnn": train_result.history}, figure_dir / "training_curves.png")
    plot_confusion_matrix_figure(
        test_metrics["targets"],
        test_metrics["predictions"],
        data_bundle.class_names,
        figure_dir / "cnn_confusion_matrix.png",
        title="CNN Confusion Matrix",
    )

    summary_row = {
        "model": "cnn",
        "params": count_parameters(model),
        "best_epoch": train_result.metrics["best_epoch"],
        "val_accuracy": round(train_result.metrics["accuracy"], 2),
        "test_accuracy": round(test_metrics["accuracy"], 2),
        "test_loss": round(test_metrics["loss"], 4),
        "training_time_sec": train_result.metrics["training_time_sec"],
        "checkpoint": str(train_result.best_checkpoint),
    }

    save_text(report_dir / "cnn_classification_report.txt", test_metrics["classification_report"])
    save_text(
        report_dir / "experiment_summary.txt",
        "\n".join(
            [
                "CIFAR-10 CNN experiment summary",
                f"Validation accuracy: {summary_row['val_accuracy']}%",
                f"Test accuracy: {summary_row['test_accuracy']}%",
                f"Best epoch: {summary_row['best_epoch']}",
                f"Trainable parameters: {summary_row['params']:,}",
                "",
                "Per-class accuracy:",
                *[f"- {name}: {score}%" for name, score in per_class_accuracy.items()],
            ]
        ),
    )
    save_results_csv(run_dir / "results_summary.csv", [summary_row])
    save_json(
        run_dir / "results_summary.json",
        {
            "device": device,
            "training_config": config,
            "result": summary_row,
            "per_class_accuracy": per_class_accuracy,
            "history": train_result.history,
        },
    )
    save_json(report_dir / "cnn_metrics.json", {"summary": summary_row, "per_class_accuracy": per_class_accuracy})

    print(
        f"CNN finished | val accuracy: {summary_row['val_accuracy']}% | "
        f"test accuracy: {summary_row['test_accuracy']}% | params: {summary_row['params']:,}"
    )


if __name__ == "__main__":
    main()
