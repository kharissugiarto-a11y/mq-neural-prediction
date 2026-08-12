"""Latih MLP klasifikasi gas dan MLP regresi PPM per kelas menggunakan NumPy."""

from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from mqml import (
    FEATURE_COLUMNS,
    MLP,
    Standardizer,
    classification_metrics,
    feature_matrix,
    group_test_split,
    parse_float,
    read_csv_rows,
    resolve_input_paths,
    safe_filename,
    stratified_validation_indices,
    write_confusion_matrix,
    write_json,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Melatih neural network klasifikasi gas dan prediksi PPM dari Rs/R0.",
    )
    parser.add_argument("--data", nargs="+", required=True, help="CSV atau pola glob CSV dataset.")
    parser.add_argument("--out", default="model", help="Folder keluaran model.")
    parser.add_argument("--epochs", type=int, default=400, help="Epoch maksimum setiap model.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--min-regression-samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def random_validation_indices(length: int, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if length < 6:
        return np.arange(length), np.array([], dtype=np.int64)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(length)
    validation_count = max(1, int(round(length * fraction)))
    return indices[validation_count:], indices[:validation_count]


def main() -> None:
    args = parse_arguments()
    if not 0.05 <= args.test_fraction <= 0.45:
        raise SystemExit("--test-fraction harus berada pada 0.05 sampai 0.45")

    paths = resolve_input_paths(args.data)
    if not paths:
        raise SystemExit("Tidak ada file dataset yang cocok dengan --data.")

    print(f"Membaca {len(paths)} file dataset...")
    source_rows = read_csv_rows(paths)
    raw_features, valid_features = feature_matrix(source_rows)

    selected_indices = [
        index
        for index, row in enumerate(source_rows)
        if valid_features[index] and str(row.get("gas_label", "")).strip()
    ]
    if len(selected_indices) < 20:
        raise SystemExit("Minimal diperlukan 20 baris dengan rasio valid dan gas_label.")

    rows = [source_rows[index] for index in selected_indices]
    features = raw_features[selected_indices]
    text_labels = np.asarray([str(row["gas_label"]).strip().upper() for row in rows])
    labels = sorted(np.unique(text_labels).tolist())
    if len(labels) < 2:
        raise SystemExit("Klasifikasi memerlukan minimal dua kelas gas.")
    label_to_index = {label: index for index, label in enumerate(labels)}
    encoded_labels = np.asarray([label_to_index[label] for label in text_labels], dtype=np.int64)
    groups = [
        str(row.get("session_id", "")).strip()
        or f"{row.get('__source_file', 'dataset')}_fallback"
        for row in rows
    ]

    train_indices, test_indices, split_warnings = group_test_split(
        text_labels,
        groups,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    local_train, local_validation = stratified_validation_indices(
        encoded_labels[train_indices],
        fraction=0.15,
        seed=args.seed + 1,
    )
    core_indices = train_indices[local_train]
    validation_indices = train_indices[local_validation]

    standardizer = Standardizer.fit(features[core_indices])
    normalized = standardizer.transform(features)
    validation_data = (
        (normalized[validation_indices], encoded_labels[validation_indices])
        if len(validation_indices)
        else None
    )

    print(
        f"Klasifikasi: {len(core_indices)} train, {len(validation_indices)} validasi, "
        f"{len(test_indices)} test, {len(labels)} kelas."
    )
    classifier = MLP(
        layer_sizes=(len(FEATURE_COLUMNS), 48, 24, len(labels)),
        task="classification",
        seed=args.seed,
        l2=1e-4,
    )
    classifier_history = classifier.fit(
        normalized[core_indices],
        encoded_labels[core_indices],
        validation=validation_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        seed=args.seed,
        verbose=not args.quiet,
    )

    predicted_test = classifier.predict(normalized[test_indices])
    classification_report = classification_metrics(
        encoded_labels[test_indices],
        predicted_test,
        labels,
    )
    print(f"Akurasi klasifikasi test: {classification_report['accuracy']:.4f}")

    output_dir = Path(args.out).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    classifier_file = "classifier.npz"
    classifier.save(output_dir / classifier_file)
    write_confusion_matrix(
        output_dir / "confusion_matrix.csv",
        classification_report["confusion_matrix"],
        labels,
    )

    regression_models: dict[str, dict[str, object]] = {}
    regression_report: dict[str, dict[str, object]] = {}
    for label in labels:
        class_indices = np.flatnonzero(text_labels == label)
        ppm_values = np.asarray([parse_float(rows[index].get("reference_ppm")) for index in class_indices])
        usable = np.isfinite(ppm_values) & (ppm_values >= 0.0)
        usable_indices = class_indices[usable]
        usable_ppm = ppm_values[usable]

        class_train_mask = np.isin(usable_indices, train_indices)
        class_test_mask = np.isin(usable_indices, test_indices)
        regression_train_indices = usable_indices[class_train_mask]
        regression_test_indices = usable_indices[class_test_mask]
        regression_train_ppm = usable_ppm[class_train_mask]
        regression_test_ppm = usable_ppm[class_test_mask]
        unique_levels = np.unique(np.round(regression_train_ppm, decimals=9))

        if len(regression_train_indices) < args.min_regression_samples or len(unique_levels) < 3:
            regression_report[label] = {
                "trained": False,
                "reason": (
                    f"Diperlukan minimal {args.min_regression_samples} sampel train dan 3 tingkat PPM; "
                    f"tersedia {len(regression_train_indices)} sampel dan {len(unique_levels)} tingkat."
                ),
            }
            continue

        regression_core_local, regression_validation_local = random_validation_indices(
            len(regression_train_indices),
            fraction=0.15,
            seed=args.seed + len(regression_models) + 10,
        )
        regression_core_indices = regression_train_indices[regression_core_local]
        regression_validation_indices = regression_train_indices[regression_validation_local]
        core_ppm = regression_train_ppm[regression_core_local]
        validation_ppm = regression_train_ppm[regression_validation_local]

        transformed_target = np.log1p(core_ppm)
        target_mean = float(transformed_target.mean())
        target_scale = float(transformed_target.std())
        if target_scale < 1e-9:
            regression_report[label] = {"trained": False, "reason": "Variasi target PPM terlalu kecil."}
            continue

        target_train = (transformed_target - target_mean) / target_scale
        validation_pair = None
        if len(regression_validation_indices):
            validation_target = (np.log1p(validation_ppm) - target_mean) / target_scale
            validation_pair = (normalized[regression_validation_indices], validation_target)

        print(
            f"Regresi {label}: {len(regression_core_indices)} train, "
            f"{len(regression_validation_indices)} validasi, {len(regression_test_indices)} test."
        )
        regressor = MLP(
            layer_sizes=(len(FEATURE_COLUMNS), 32, 16, 1),
            task="regression",
            seed=args.seed + len(regression_models) + 100,
            l2=2e-4,
        )
        regressor_history = regressor.fit(
            normalized[regression_core_indices],
            target_train,
            validation=validation_pair,
            epochs=args.epochs,
            batch_size=min(args.batch_size, 32),
            learning_rate=args.learning_rate,
            patience=args.patience,
            seed=args.seed + len(regression_models) + 100,
            verbose=not args.quiet,
        )

        filename = f"regressor_{safe_filename(label)}.npz"
        regressor.save(output_dir / filename)
        model_info: dict[str, object] = {
            "file": filename,
            "target_transform": "standardized_log1p_ppm",
            "target_mean": target_mean,
            "target_scale": target_scale,
            "train_samples": int(len(regression_train_indices)),
            "unique_ppm_levels": int(len(unique_levels)),
            "training_ppm_min": float(regression_train_ppm.min()),
            "training_ppm_max": float(regression_train_ppm.max()),
        }
        regression_models[label] = model_info

        metrics: dict[str, object] = {
            "trained": True,
            "best_epoch": regressor_history["best_epoch"],
            "best_validation_loss": regressor_history["best_validation_loss"],
        }
        if len(regression_test_indices):
            standardized_prediction = regressor.predict(normalized[regression_test_indices])
            ppm_prediction = np.maximum(0.0, np.expm1(standardized_prediction * target_scale + target_mean))
            errors = ppm_prediction - regression_test_ppm
            metrics.update({
                "test_samples": int(len(regression_test_indices)),
                "mae_ppm": float(np.mean(np.abs(errors))),
                "rmse_ppm": float(math.sqrt(float(np.mean(errors ** 2)))),
            })
            print(f"  MAE test {label}: {metrics['mae_ppm']:.4f} ppm")
        else:
            metrics.update({"test_samples": 0, "mae_ppm": None, "rmse_ppm": None})
        regression_report[label] = metrics

    metadata = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feature_columns": list(FEATURE_COLUMNS),
        "feature_transform": "natural_log_rs_over_r0_then_standardize",
        "standardizer": standardizer.to_json(),
        "labels": labels,
        "classifier_file": classifier_file,
        "classifier_architecture": list(classifier.layer_sizes),
        "regression_models": regression_models,
        "warning": "Prediksi hanya berlaku pada sensor, kelas, rentang PPM, dan kondisi yang terwakili dalam data pelatihan.",
    }
    report = {
        "source_files": [str(path) for path in paths],
        "rows_read": len(source_rows),
        "rows_used": len(rows),
        "train_rows": int(len(train_indices)),
        "test_rows": int(len(test_indices)),
        "split_warnings": split_warnings,
        "classification": {
            **classification_report,
            "best_epoch": classifier_history["best_epoch"],
            "best_validation_loss": classifier_history["best_validation_loss"],
        },
        "regression": regression_report,
    }
    write_json(output_dir / "metadata.json", metadata)
    write_json(output_dir / "training_report.json", report)

    for warning in split_warnings:
        print(f"PERINGATAN: {warning}")
    print(f"Model tersimpan di: {output_dir}")


if __name__ == "__main__":
    main()

