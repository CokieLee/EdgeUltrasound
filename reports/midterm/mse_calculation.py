"""Calculate midterm validation MSEs using only the Python standard library.

Run from any directory: python <repo>/reports/midterm/mse_calculation.py
Defaults use the repeated v2 CNN (25 epochs) and SNN (75 epochs) experiments.
Each run contributes one final-epoch and one best-epoch validation MSE;
standard deviations are sample standard deviations across independent runs.

The SoS predictions are read as stored, without refitting. The existing CSV
regression was fitted on all rats, so this is not a held-out regression fit.
Network losses retain their logged units because target normalization is not
recorded in the result files. If targets were divided by 100 during training,
compare those losses with the fraction-scale SoS MSE (percent-scale / 10000).
"""

import argparse
import csv
import json
import math
from pathlib import Path
import re
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[2]
VAL_RAT_IDS = (6, 10, 15, 18)
PREDICTION_COLUMN = "FSA_sos_linear_to_fat"


def regression_mse(data_path):
    """Return percent-scale MSE for the four validation rats, one row per rat."""
    squared_errors = {}
    with data_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            rat_id = int(row["rat_index"])
            if rat_id not in VAL_RAT_IDS:
                continue
            if rat_id in squared_errors:
                raise ValueError(f"Duplicate validation rat {rat_id} in {data_path}")
            prediction = float(row[PREDICTION_COLUMN])
            target = float(row["liver_fat_percent"])
            if not all(math.isfinite(value) for value in (prediction, target)):
                raise ValueError(f"Non-finite measurement for rat {rat_id}")
            squared_errors[rat_id] = (prediction - target) ** 2
    missing = set(VAL_RAT_IDS) - squared_errors.keys()
    if missing:
        raise ValueError(f"Missing validation rats in {data_path}: {sorted(missing)}")
    return mean(squared_errors.values())


def network_runs(results_dir, spiking, epochs):
    """Select v2 training runs with the exact validation split and epoch count."""
    runs = {}
    # Top-level training logs only; benchmark subdirectories repeat evaluations.
    for path in sorted(results_dir.glob("*.json")):
        with path.open(encoding="utf-8") as handle:
            run = json.load(handle)
        notes = run.get("notes", "")
        if not re.search(r"\bv2\b", notes):
            continue
        split = re.search(r"\bval_ids\s*=\s*\[([\d,\s]+)\]", notes)
        if split is None:
            continue
        rat_ids = [int(value.strip()) for value in split[1].split(",")]
        if sorted(rat_ids) != sorted(VAL_RAT_IDS):
            continue
        if run.get("init_params", {}).get("spiking") is not spiking:
            continue
        if run.get("epochs") != epochs:
            continue
        history = run.get("val_history", [])
        if len(history) != epochs or not all(
            isinstance(value, (int, float)) and math.isfinite(value) and value >= 0
            for value in history
        ):
            raise ValueError(f"Invalid or incomplete validation history: {path}")
        runs[path.name] = run
    if len(runs) < 2:
        model = "SNN" if spiking else "CNN"
        raise ValueError(f"Need at least two matching {model} runs at {epochs} epochs")
    return runs


def network_mses(results_dir, spiking, epochs):
    runs = network_runs(results_dir, spiking, epochs)
    histories = [run["val_history"] for run in runs.values()]
    return [history[-1] for history in histories], [min(history) for history in histories]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path,
                        default=ROOT / "Training_Data" / "training_targets.csv")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--cnn-epochs", type=int, default=25)
    parser.add_argument("--snn-epochs", type=int, default=75)
    args = parser.parse_args()

    sos_mse = regression_mse(args.targets)
    networks = [
        (name, epochs, *network_mses(args.results_dir, spiking, epochs))
        for name, spiking, epochs in (
            ("CNN", False, args.cnn_epochs), ("SNN", True, args.snn_epochs)
        )
    ]
    print(f"Validation rat IDs: {list(VAL_RAT_IDS)} (CSV rat_index)")
    print(f"SoS regression MSE: {sos_mse:.10f} (percentage points)^2")
    print(f"SoS regression MSE: {sos_mse / 10000:.10f} (fat fraction)^2")
    print("SoS uses stored predictions from a regression fitted on all rats.")
    print("\nNetwork MSE in logged units; sample stdev across runs (ddof=1):")
    for name, epochs, final_mses, best_mses in networks:
        print(f"{name}: {len(final_mses)} v2 runs, {epochs} epochs")
        print(f"  Final epoch: mean={mean(final_mses):.10f}, stdev={stdev(final_mses):.10f}")
        print(f"  Best epoch:  mean={mean(best_mses):.10f}, stdev={stdev(best_mses):.10f}")
    print("Target normalization is absent from logs. If targets were /100, use the fraction-scale SoS MSE.")


if __name__ == "__main__":
    main()
