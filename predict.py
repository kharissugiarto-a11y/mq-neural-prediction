"""Gunakan model tersimpan untuk klasifikasi gas dan prediksi PPM CSV baru."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from mqml import MLP, Standardizer, feature_matrix, read_csv_rows


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prediksi kelas gas dan PPM dari CSV Rs/R0.")
    parser.add_argument("--model", required=True, help="Folder model hasil train.py.")
    parser.add_argument("--input", required=True, help="CSV yang akan diprediksi.")
    parser.add_argument("--output", required=True, help="Lokasi CSV hasil prediksi.")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    model_dir = Path(args.model).resolve()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    metadata_path = model_dir / "metadata.json"
    if not metadata_path.exists():
        raise SystemExit(f"metadata.json tidak ditemukan di {model_dir}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    labels: list[str] = metadata["labels"]
    standardizer = Standardizer.from_json(metadata["standardizer"])
    classifier = MLP.load(model_dir / metadata["classifier_file"])

    rows = read_csv_rows([input_path])
    features, valid = feature_matrix(rows)
    normalized = np.zeros_like(features)
    normalized[valid] = standardizer.transform(features[valid])

    predicted_labels: list[str | None] = [None] * len(rows)
    confidences: list[float | None] = [None] * len(rows)
    predicted_ppm: list[float | None] = [None] * len(rows)
    ppm_available: list[bool] = [False] * len(rows)
    outside_range: list[bool | None] = [None] * len(rows)

    valid_indices = np.flatnonzero(valid)
    if len(valid_indices):
        probabilities = classifier.predict_proba(normalized[valid_indices])
        class_indices = np.argmax(probabilities, axis=1)
        for local_index, row_index in enumerate(valid_indices):
            predicted_labels[row_index] = labels[int(class_indices[local_index])]
            confidences[row_index] = float(probabilities[local_index, class_indices[local_index]])

        regression_models: dict[str, dict[str, object]] = metadata.get("regression_models", {})
        for label, model_info in regression_models.items():
            matching = np.asarray(
                [index for index in valid_indices if predicted_labels[index] == label],
                dtype=np.int64,
            )
            if not len(matching):
                continue
            regressor = MLP.load(model_dir / str(model_info["file"]))
            standardized = regressor.predict(normalized[matching])
            target_mean = float(model_info["target_mean"])
            target_scale = float(model_info["target_scale"])
            ppm_values = np.maximum(0.0, np.expm1(standardized * target_scale + target_mean))
            training_min = float(model_info["training_ppm_min"])
            training_max = float(model_info["training_ppm_max"])
            for row_index, ppm in zip(matching, ppm_values):
                predicted_ppm[int(row_index)] = float(ppm)
                ppm_available[int(row_index)] = True
                outside_range[int(row_index)] = bool(ppm < training_min or ppm > training_max)

    source_columns = [column for column in rows[0].keys() if column != "__source_file"]
    result_columns = [
        *source_columns,
        "predicted_gas",
        "classification_confidence",
        "predicted_ppm",
        "ppm_model_available",
        "ppm_outside_training_range",
        "prediction_error",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=result_columns)
        writer.writeheader()
        for index, row in enumerate(rows):
            result = {column: row.get(column, "") for column in source_columns}
            result.update({
                "predicted_gas": predicted_labels[index] or "",
                "classification_confidence": (
                    f"{confidences[index]:.6f}" if confidences[index] is not None else ""
                ),
                "predicted_ppm": (
                    f"{predicted_ppm[index]:.6f}" if predicted_ppm[index] is not None else ""
                ),
                "ppm_model_available": "true" if ppm_available[index] else "false",
                "ppm_outside_training_range": (
                    "true" if outside_range[index] is True else "false" if outside_range[index] is False else ""
                ),
                "prediction_error": "" if valid[index] else "Rasio sensor kosong, nol, atau tidak valid",
            })
            writer.writerow(result)

    valid_count = int(valid.sum())
    ppm_count = sum(ppm_available)
    print(f"Prediksi klasifikasi: {valid_count}/{len(rows)} baris")
    print(f"Prediksi PPM tersedia: {ppm_count}/{len(rows)} baris")
    print(f"Hasil tersimpan di: {output_path}")


if __name__ == "__main__":
    main()

