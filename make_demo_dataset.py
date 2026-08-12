"""Buat dataset sintetis untuk smoke test pipeline. Bukan data penelitian."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np


SENSORS = ("mq6", "mq2", "mq135", "mq3", "mq131")
CLEAN_RATIO = np.asarray([10.0, 9.83, 3.60, 60.0, 1.0])
R0 = np.asarray([0.18, 0.22, 0.35, 0.03, 0.8])

# Besar nilai menunjukkan sensor yang lebih responsif terhadap kelas sintetis.
SIGNATURES = {
    "CLEAN_AIR": np.asarray([0.00, 0.00, 0.00, 0.00, 0.00]),
    "METHANE": np.asarray([0.72, 0.28, 0.08, 0.04, 0.00]),
    "LPG": np.asarray([0.42, 0.78, 0.08, 0.04, 0.00]),
    "SMOKE": np.asarray([0.12, 0.62, 0.25, 0.10, 0.02]),
    "AMMONIA": np.asarray([0.05, 0.10, 0.75, 0.06, 0.02]),
    "TOLUENE": np.asarray([0.06, 0.18, 0.66, 0.32, 0.01]),
    "ETHANOL": np.asarray([0.04, 0.22, 0.28, 0.82, 0.01]),
    "OZONE": np.asarray([0.00, 0.02, 0.08, 0.00, -0.85]),
    "SULFIDE": np.asarray([0.04, 0.20, 0.70, 0.08, 0.03]),
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--sessions-per-class", type=int, default=5)
    parser.add_argument("--samples-per-level", type=int, default=18)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    concentrations = (10.0, 40.0, 120.0, 350.0)
    start_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows: list[dict[str, object]] = []
    sequence = 0

    for label, signature in SIGNATURES.items():
        for session in range(args.sessions_per_class):
            session_id = f"DEMO_{label}_{session + 1:02d}"
            session_shift = rng.normal(0.0, 0.025, len(SENSORS))
            levels = (0.0,) if label == "CLEAN_AIR" else concentrations
            for ppm in levels:
                strength = np.log1p(ppm) / np.log1p(max(concentrations)) if ppm else 0.0
                for sample in range(args.samples_per_level):
                    sequence += 1
                    noise = rng.normal(0.0, 0.018, len(SENSORS))
                    if label == "OZONE":
                        ratio = CLEAN_RATIO * np.exp(-signature * strength + session_shift + noise)
                    else:
                        ratio = CLEAN_RATIO * np.exp(-signature * strength + session_shift + noise)
                    ratio = np.maximum(ratio, 1e-5)
                    rs_factor = ratio * R0
                    adc = np.clip(np.rint(1023.0 / (rs_factor + 1.0)), 1, 1022).astype(int)
                    timestamp = start_time + timedelta(seconds=sequence)
                    row: dict[str, object] = {
                        "session_id": session_id,
                        "timestamp_iso": timestamp.isoformat(),
                        "sample_index": sample + 1,
                        "recording_elapsed_ms": sample * 1000,
                        "arduino_ms": sequence * 1000,
                        "gas_label": label,
                        "reference_ppm": ppm,
                        "temperature_c": 25.0 + rng.normal(0.0, 0.3),
                        "humidity_percent": 55.0 + rng.normal(0.0, 0.8),
                        "notes": "SYNTHETIC_PIPELINE_TEST_ONLY",
                        "calibrated": "true",
                        "mq131_model": "LOW",
                    }
                    for index, key in enumerate(SENSORS):
                        row[f"adc_{key}"] = int(adc[index])
                        row[f"r0_{key}"] = float(R0[index])
                        row[f"ratio_{key}"] = float(ratio[index])
                    rows.append(row)

    fieldnames = list(rows[0].keys())
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Dataset sintetis {len(rows)} baris dibuat di: {output}")
    print("PERINGATAN: dataset ini hanya untuk menguji program, bukan untuk hasil penelitian.")


if __name__ == "__main__":
    main()

