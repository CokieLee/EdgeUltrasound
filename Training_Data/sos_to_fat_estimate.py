"""View liver-fat estimates from local FSA speed-of-sound measurements."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt


DATA_PATH = Path(__file__).with_name("training_targets.csv")
LINEAR_FAT_COLUMN = "FSA_sos_linear_to_fat"
# No longer using but can be plotted to compare against paper:
SOS_TO_FAT_SLOPE = -1.63  # m/s per percentage point of fat, in the paper's figure
ZERO_FAT_SOS = 1575  # m/s at 0% fat, from the papers figure, not exact, just eyeballed


def load_measurements(data_path=DATA_PATH):
    measurements = []
    with data_path.open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            if not row["local_sos_fsa_mps"] or not row["liver_fat_percent"]:
                continue
            local_sos = float(row["local_sos_fsa_mps"])
            true_fat = float(row["liver_fat_percent"])
            estimated_fat = (local_sos - ZERO_FAT_SOS) / SOS_TO_FAT_SLOPE
            measurements.append((row["rat_id"], local_sos, true_fat, estimated_fat))
    return measurements


def plot_estimates(measurements):
    rat_ids, local_sos, true_fat, estimated_fat = zip(*measurements)

    figure, (sos_axis, comparison_axis) = plt.subplots(1, 2, figsize=(13, 5))

    sos_axis.scatter(local_sos, estimated_fat)
    for rat_id, sos, estimate in zip(rat_ids, local_sos, estimated_fat):
        sos_axis.annotate(rat_id, (sos, estimate), xytext=(4, 4), textcoords="offset points")
    sos_axis.set_title("Local FSA SoS vs. Estimated Liver Fat")
    sos_axis.set_xlabel("Local FSA speed of sound (m/s)")
    sos_axis.set_ylabel("Estimated liver fat (%)")
    sos_axis.grid(alpha=0.3)

    comparison_axis.scatter(true_fat, estimated_fat)
    limits = [
        min(min(true_fat), min(estimated_fat)),
        max(max(true_fat), max(estimated_fat)),
    ]
    comparison_axis.plot(limits, limits, "k--", label="x = y")
    comparison_axis.set_title("True vs. Estimated Liver Fat")
    comparison_axis.set_xlabel("True liver fat (%)")
    comparison_axis.set_ylabel("Estimated liver fat (%)")
    comparison_axis.legend()
    comparison_axis.grid(alpha=0.3)

    figure.tight_layout()
    return figure


def linear_regression(measurements):
    _, local_sos, true_fat, _ = zip(*measurements)
    mean_sos = sum(local_sos) / len(local_sos)
    mean_fat = sum(true_fat) / len(true_fat)
    slope = sum(
        (sos - mean_sos) * (fat - mean_fat)
        for sos, fat in zip(local_sos, true_fat)
    ) / sum((sos - mean_sos) ** 2 for sos in local_sos)
    intercept = mean_fat - slope * mean_sos
    fitted_fat = [intercept + slope * sos for sos in local_sos]
    total_sum_squares = sum((fat - mean_fat) ** 2 for fat in true_fat)
    residual_sum_squares = sum(
        (fat - fitted) ** 2 for fat, fitted in zip(true_fat, fitted_fat)
    )
    mean_squared_error = residual_sum_squares / len(true_fat)
    r_squared = 1 - residual_sum_squares / total_sum_squares
    return slope, intercept, r_squared, mean_squared_error


def update_training_targets(data_path, slope, intercept):
    with data_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    if LINEAR_FAT_COLUMN not in fieldnames:
        fieldnames.append(LINEAR_FAT_COLUMN)

    for row in rows:
        local_sos = row["local_sos_fsa_mps"]
        row[LINEAR_FAT_COLUMN] = (
            f"{intercept + slope * float(local_sos):.6f}" if local_sos else ""
        )

    with data_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_linear_regression(measurements, slope, intercept, r_squared):
    _, local_sos, true_fat, _ = zip(*measurements)

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.scatter(local_sos, true_fat, label="Rat measurements")
    line_sos = [min(local_sos), max(local_sos)]
    axis.plot(
        line_sos,
        [intercept + slope * sos for sos in line_sos],
        "r-",
        label=f"Fit: fat = {slope:.3f} SoS + {intercept:.1f}",
    )
    axis.set_title(f"Local FSA SoS vs. Liver Fat ($R^2 = {r_squared:.3f}$)")
    axis.set_xlabel("Local FSA speed of sound (m/s)")
    axis.set_ylabel("True liver fat (%)")
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()
    return figure


if __name__ == "__main__":
    measurements = load_measurements()
    slope, intercept, r_squared, mean_squared_error = linear_regression(measurements)
    print(f"Linear regression MSE: {mean_squared_error:.6f} (% fat)^2")
    update_training_targets(DATA_PATH, slope, intercept)
    plot_estimates(measurements)
    plot_linear_regression(measurements, slope, intercept, r_squared)
    plt.show()