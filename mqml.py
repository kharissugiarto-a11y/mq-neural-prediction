"""Utilitas NumPy untuk klasifikasi gas dan regresi PPM sensor MQ."""

from __future__ import annotations

import csv
import glob
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


SENSOR_KEYS = ("mq6", "mq2", "mq135", "mq3", "mq131")
FEATURE_COLUMNS = tuple(f"ratio_{key}" for key in SENSOR_KEYS)


def resolve_input_paths(patterns: Sequence[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = [Path(item) for item in glob.glob(pattern)]
        if matches:
            paths.extend(matches)
        else:
            candidate = Path(pattern)
            if candidate.exists():
                paths.append(candidate)
    return sorted(set(path.resolve() for path in paths if path.is_file()))


def detect_delimiter(path: Path) -> str:
    sample = path.read_text(encoding="utf-8-sig", errors="replace")[:4096]
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
    except csv.Error:
        return ","


def read_csv_rows(paths: Sequence[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=detect_delimiter(path))
            if not reader.fieldnames:
                raise ValueError(f"CSV tidak memiliki header: {path}")
            missing = [column for column in FEATURE_COLUMNS if column not in reader.fieldnames]
            if missing:
                raise ValueError(f"Kolom {', '.join(missing)} tidak ditemukan pada {path}")
            for row in reader:
                row["__source_file"] = path.name
                rows.append(row)
    if not rows:
        raise ValueError("Tidak ada baris data yang ditemukan.")
    return rows


def parse_float(value: object) -> float:
    if value is None:
        return math.nan
    text = str(value).strip()
    if not text:
        return math.nan
    if "," in text and "." not in text:
        text = text.replace(",", ".")
    try:
        number = float(text)
    except ValueError:
        return math.nan
    return number if math.isfinite(number) else math.nan


def feature_matrix(rows: Sequence[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.full((len(rows), len(FEATURE_COLUMNS)), np.nan, dtype=np.float64)
    for row_index, row in enumerate(rows):
        for column_index, column in enumerate(FEATURE_COLUMNS):
            value = parse_float(row.get(column))
            if math.isfinite(value) and value > 0:
                matrix[row_index, column_index] = math.log(value)
    valid = np.all(np.isfinite(matrix), axis=1)
    return matrix, valid


@dataclass
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "Standardizer":
        mean = values.mean(axis=0)
        scale = values.std(axis=0)
        scale = np.where(scale < 1e-9, 1.0, scale)
        return cls(mean=mean, scale=scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.scale

    def to_json(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_json(cls, payload: dict[str, list[float]]) -> "Standardizer":
        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float64),
            scale=np.asarray(payload["scale"], dtype=np.float64),
        )


class MLP:
    """Jaringan saraf feed-forward kecil dengan ReLU dan optimizer Adam."""

    def __init__(
        self,
        layer_sizes: Sequence[int],
        task: str,
        seed: int = 42,
        l2: float = 1e-4,
    ) -> None:
        if task not in {"classification", "regression"}:
            raise ValueError("task harus classification atau regression")
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes minimal berisi input dan output")
        self.layer_sizes = tuple(int(size) for size in layer_sizes)
        self.task = task
        self.l2 = float(l2)
        rng = np.random.default_rng(seed)
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        for input_size, output_size in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            scale = math.sqrt(2.0 / input_size)
            self.weights.append(rng.normal(0.0, scale, (input_size, output_size)))
            self.biases.append(np.zeros((1, output_size), dtype=np.float64))

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

    def _forward(self, inputs: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        activations = [inputs]
        preactivations: list[np.ndarray] = []
        current = inputs
        for layer_index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            z_value = current @ weight + bias
            preactivations.append(z_value)
            if layer_index < len(self.weights) - 1:
                current = np.maximum(z_value, 0.0)
            else:
                current = z_value
            activations.append(current)
        return activations, preactivations

    def _loss_and_gradients(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
    ) -> tuple[float, list[np.ndarray], list[np.ndarray]]:
        activations, preactivations = self._forward(inputs)
        output = activations[-1]
        sample_count = max(1, inputs.shape[0])

        if self.task == "classification":
            probabilities = self._softmax(output)
            target_indices = targets.astype(np.int64).reshape(-1)
            loss = -np.log(np.clip(probabilities[np.arange(sample_count), target_indices], 1e-12, 1.0)).mean()
            delta = probabilities
            delta[np.arange(sample_count), target_indices] -= 1.0
            delta /= sample_count
        else:
            expected = targets.reshape(output.shape)
            difference = output - expected
            loss = float(np.mean(difference ** 2))
            delta = 2.0 * difference / sample_count

        loss += 0.5 * self.l2 * sum(float(np.sum(weight ** 2)) for weight in self.weights)
        weight_gradients: list[np.ndarray] = [np.empty(0)] * len(self.weights)
        bias_gradients: list[np.ndarray] = [np.empty(0)] * len(self.biases)

        for layer_index in reversed(range(len(self.weights))):
            weight_gradients[layer_index] = activations[layer_index].T @ delta + self.l2 * self.weights[layer_index]
            bias_gradients[layer_index] = delta.sum(axis=0, keepdims=True)
            if layer_index > 0:
                delta = (delta @ self.weights[layer_index].T) * (preactivations[layer_index - 1] > 0.0)

        return float(loss), weight_gradients, bias_gradients

    def loss(self, inputs: np.ndarray, targets: np.ndarray) -> float:
        activations, _ = self._forward(inputs)
        output = activations[-1]
        if self.task == "classification":
            probabilities = self._softmax(output)
            indices = targets.astype(np.int64).reshape(-1)
            base = -np.log(np.clip(probabilities[np.arange(len(indices)), indices], 1e-12, 1.0)).mean()
        else:
            base = np.mean((output - targets.reshape(output.shape)) ** 2)
        penalty = 0.5 * self.l2 * sum(float(np.sum(weight ** 2)) for weight in self.weights)
        return float(base + penalty)

    def fit(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        validation: tuple[np.ndarray, np.ndarray] | None = None,
        epochs: int = 400,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        patience: int = 35,
        seed: int = 42,
        verbose: bool = True,
    ) -> dict[str, object]:
        rng = np.random.default_rng(seed)
        moment_w = [np.zeros_like(weight) for weight in self.weights]
        velocity_w = [np.zeros_like(weight) for weight in self.weights]
        moment_b = [np.zeros_like(bias) for bias in self.biases]
        velocity_b = [np.zeros_like(bias) for bias in self.biases]
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        step = 0
        best_loss = math.inf
        best_weights = [weight.copy() for weight in self.weights]
        best_biases = [bias.copy() for bias in self.biases]
        best_epoch = 0
        stale_epochs = 0
        history: list[dict[str, float]] = []

        for epoch in range(1, epochs + 1):
            permutation = rng.permutation(inputs.shape[0])
            epoch_losses: list[float] = []
            for start in range(0, inputs.shape[0], batch_size):
                indices = permutation[start:start + batch_size]
                batch_x = inputs[indices]
                batch_y = targets[indices]
                batch_loss, gradients_w, gradients_b = self._loss_and_gradients(batch_x, batch_y)
                epoch_losses.append(batch_loss)
                step += 1

                for index in range(len(self.weights)):
                    moment_w[index] = beta1 * moment_w[index] + (1.0 - beta1) * gradients_w[index]
                    velocity_w[index] = beta2 * velocity_w[index] + (1.0 - beta2) * (gradients_w[index] ** 2)
                    moment_b[index] = beta1 * moment_b[index] + (1.0 - beta1) * gradients_b[index]
                    velocity_b[index] = beta2 * velocity_b[index] + (1.0 - beta2) * (gradients_b[index] ** 2)

                    corrected_mw = moment_w[index] / (1.0 - beta1 ** step)
                    corrected_vw = velocity_w[index] / (1.0 - beta2 ** step)
                    corrected_mb = moment_b[index] / (1.0 - beta1 ** step)
                    corrected_vb = velocity_b[index] / (1.0 - beta2 ** step)
                    self.weights[index] -= learning_rate * corrected_mw / (np.sqrt(corrected_vw) + epsilon)
                    self.biases[index] -= learning_rate * corrected_mb / (np.sqrt(corrected_vb) + epsilon)

            training_loss = float(np.mean(epoch_losses))
            monitored_loss = self.loss(*validation) if validation is not None else training_loss
            history.append({"epoch": epoch, "train_loss": training_loss, "validation_loss": monitored_loss})

            if monitored_loss < best_loss - 1e-7:
                best_loss = monitored_loss
                best_epoch = epoch
                stale_epochs = 0
                best_weights = [weight.copy() for weight in self.weights]
                best_biases = [bias.copy() for bias in self.biases]
            else:
                stale_epochs += 1

            if verbose and (epoch == 1 or epoch % 25 == 0):
                print(f"  epoch {epoch:4d} | train={training_loss:.5f} | valid={monitored_loss:.5f}")
            if stale_epochs >= patience:
                break

        self.weights = best_weights
        self.biases = best_biases
        return {
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "epochs_completed": len(history),
            "history": history,
        }

    def predict_proba(self, inputs: np.ndarray) -> np.ndarray:
        if self.task != "classification":
            raise ValueError("predict_proba hanya untuk classification")
        output = self._forward(inputs)[0][-1]
        return self._softmax(output)

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        output = self._forward(inputs)[0][-1]
        if self.task == "classification":
            return np.argmax(output, axis=1)
        return output.reshape(-1)

    def save(self, path: Path) -> None:
        payload: dict[str, np.ndarray] = {
            "layer_sizes": np.asarray(self.layer_sizes, dtype=np.int64),
            "task": np.asarray(self.task),
            "l2": np.asarray(self.l2, dtype=np.float64),
        }
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            payload[f"weight_{index}"] = weight
            payload[f"bias_{index}"] = bias
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: Path) -> "MLP":
        with np.load(path, allow_pickle=False) as payload:
            model = cls(
                layer_sizes=payload["layer_sizes"].tolist(),
                task=str(payload["task"].item()),
                l2=float(payload["l2"].item()),
            )
            for index in range(len(model.weights)):
                model.weights[index] = payload[f"weight_{index}"].copy()
                model.biases[index] = payload[f"bias_{index}"].copy()
        return model


def safe_filename(label: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return cleaned or "unknown"


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def stratified_validation_indices(
    labels: np.ndarray,
    fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    validation_indices: list[int] = []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        validation_count = max(1, int(round(len(indices) * fraction))) if len(indices) >= 5 else 0
        validation_indices.extend(indices[:validation_count].tolist())
        train_indices.extend(indices[validation_count:].tolist())
    if not validation_indices:
        return np.arange(len(labels)), np.array([], dtype=np.int64)
    return np.asarray(train_indices, dtype=np.int64), np.asarray(validation_indices, dtype=np.int64)


def group_test_split(
    labels: Sequence[str],
    groups: Sequence[str],
    test_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Pisahkan sesi per kelas; fallback baris acak jika suatu kelas hanya punya satu sesi."""
    rng = np.random.default_rng(seed)
    labels_array = np.asarray(labels)
    groups_array = np.asarray(groups)
    test_mask = np.zeros(len(labels_array), dtype=bool)
    warnings: list[str] = []

    for label in np.unique(labels_array):
        class_indices = np.flatnonzero(labels_array == label)
        class_groups = np.unique(groups_array[class_indices])
        if len(class_groups) >= 2:
            shuffled = class_groups.copy()
            rng.shuffle(shuffled)
            test_group_count = max(1, int(round(len(shuffled) * test_fraction)))
            chosen = set(shuffled[:test_group_count].tolist())
            test_mask[class_indices] = np.isin(groups_array[class_indices], list(chosen))
        else:
            shuffled_indices = class_indices.copy()
            rng.shuffle(shuffled_indices)
            test_count = max(1, int(round(len(shuffled_indices) * test_fraction))) if len(shuffled_indices) >= 5 else 0
            test_mask[shuffled_indices[:test_count]] = True
            warnings.append(
                f"Kelas {label} hanya memiliki satu sesi; pemisahan baris digunakan dan dapat membuat evaluasi terlalu optimistis."
            )

    train_indices = np.flatnonzero(~test_mask)
    test_indices = np.flatnonzero(test_mask)
    if not len(train_indices) or not len(test_indices):
        raise ValueError("Dataset tidak cukup untuk membuat bagian train dan test.")
    return train_indices, test_indices, warnings


def classification_metrics(
    true_values: np.ndarray,
    predicted_values: np.ndarray,
    labels: Sequence[str],
) -> dict[str, object]:
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for true_value, predicted_value in zip(true_values, predicted_values):
        matrix[int(true_value), int(predicted_value)] += 1

    per_class: dict[str, dict[str, float | int]] = {}
    for index, label in enumerate(labels):
        true_positive = matrix[index, index]
        false_positive = matrix[:, index].sum() - true_positive
        false_negative = matrix[index, :].sum() - true_positive
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        per_class[label] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(matrix[index, :].sum()),
        }
    return {
        "accuracy": float(np.mean(true_values == predicted_values)),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


def write_confusion_matrix(path: Path, matrix: Sequence[Sequence[int]], labels: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual/predicted", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *row])

