"""Training and evaluation helpers."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import classification_report
from torch import nn
from tqdm import tqdm


@dataclass(slots=True)
class TrainingResult:
    model_name: str
    history: dict[str, list[float]]
    best_checkpoint: Path
    metrics: dict[str, Any]


class ModelTrainer:
    """Reusable trainer for classification experiments."""

    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        device: str,
        output_dir: str | Path,
        config: dict[str, Any],
    ) -> None:
        self.model = model
        self.model_name = model_name
        self.device = torch.device(device)
        self.model.to(self.device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.output_dir / f"{model_name}_best.pt"
        self.config = config

        label_smoothing = float(config.get("label_smoothing", 0.0))
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.optimizer = self._build_optimizer(config)
        self.scheduler = self._build_scheduler(config)

        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "lr": [],
        }

    def _build_optimizer(self, config: dict[str, Any]):
        optimizer_name = config.get("optimizer", "adamw").lower()
        lr = float(config.get("learning_rate", 1e-3))
        weight_decay = float(config.get("weight_decay", 1e-4))

        if optimizer_name == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=float(config.get("momentum", 0.9)),
                nesterov=True,
                weight_decay=weight_decay,
            )
        if optimizer_name == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        return torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def _build_scheduler(self, config: dict[str, Any]):
        scheduler_name = config.get("scheduler", "cosine").lower()
        epochs = int(config.get("epochs", 30))
        if scheduler_name == "multistep":
            milestones = config.get("milestones", [epochs // 2, int(epochs * 0.75)])
            return torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=0.1)
        if scheduler_name == "none":
            return None
        return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)

    def _run_one_epoch(self, loader, training: bool) -> tuple[float, float]:
        self.model.train(training)
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with tqdm(loader, leave=True, desc="train" if training else "eval") as progress:
            for inputs, targets in progress:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                if training:
                    self.optimizer.zero_grad(set_to_none=True)

                with torch.set_grad_enabled(training):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

                if training:
                    loss.backward()
                    if self.config.get("grad_clip"):
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.config["grad_clip"]))
                    self.optimizer.step()

                batch_size = targets.size(0)
                total_loss += float(loss.item()) * batch_size
                total_correct += (outputs.argmax(dim=1) == targets).sum().item()
                total_samples += batch_size

                progress.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100 * total_correct / total_samples:.2f}%")

        return total_loss / total_samples, 100.0 * total_correct / total_samples

    def train(self, train_loader, val_loader) -> TrainingResult:
        best_state = None
        best_val_acc = -1.0
        best_epoch = 0
        patience = int(self.config.get("early_stopping_patience", 10))
        patience_counter = 0
        start_time = time.time()

        for epoch in range(1, int(self.config.get("epochs", 30)) + 1):
            train_loss, train_acc = self._run_one_epoch(train_loader, training=True)
            val_loss, val_acc = self._run_one_epoch(val_loader, training=False)

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(self.optimizer.param_groups[0]["lr"])

            if self.scheduler is not None:
                self.scheduler.step()

            print(
                f"[{self.model_name}] epoch {epoch:02d} | "
                f"train loss {train_loss:.4f} acc {train_acc:.2f}% | "
                f"val loss {val_loss:.4f} acc {val_acc:.2f}%"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                patience_counter = 0
                best_state = copy.deepcopy(self.model.state_dict())
                torch.save(best_state, self.checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"[{self.model_name}] early stopping at epoch {epoch}.")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        metrics = self.evaluate(val_loader, class_names=None)
        metrics["best_epoch"] = best_epoch
        metrics["training_time_sec"] = round(time.time() - start_time, 2)

        return TrainingResult(
            model_name=self.model_name,
            history=self.history,
            best_checkpoint=self.checkpoint_path,
            metrics=metrics,
        )

    def evaluate(self, loader, class_names: list[str] | None):
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        predictions: list[int] = []
        targets_all: list[int] = []

        with torch.no_grad():
            for inputs, targets in tqdm(loader, leave=False, desc="test"):
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                preds = outputs.argmax(dim=1)

                batch_size = targets.size(0)
                total_loss += float(loss.item()) * batch_size
                total_correct += (preds == targets).sum().item()
                total_samples += batch_size

                predictions.extend(preds.cpu().numpy().tolist())
                targets_all.extend(targets.cpu().numpy().tolist())

        metrics = {
            "loss": total_loss / total_samples,
            "accuracy": 100.0 * total_correct / total_samples,
            "predictions": np.array(predictions, dtype=np.int64),
            "targets": np.array(targets_all, dtype=np.int64),
        }
        if class_names is not None:
            metrics["classification_report"] = classification_report(
                targets_all,
                predictions,
                target_names=class_names,
                digits=4,
                zero_division=0,
            )
        return metrics
