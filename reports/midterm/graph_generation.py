"""Generate CNN/SNN MSE curves and spike benchmarks with standard deviations.

Run: python reports/midterm/graph_generation.py
Requires: matplotlib (or install the project's `reports` extra).
Network targets are assumed to be fat fractions for the regression comparison.
CNN and SNN cohorts match mse_calculation.py: 25 and 75 epochs by default.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter

if __package__:
    from .benchmark_calculation import load_groups, metric_values, summarize_spikes
    from .mse_calculation import ROOT, VAL_RAT_IDS, network_runs, regression_mse
else:
    from benchmark_calculation import load_groups, metric_values, summarize_spikes
    from mse_calculation import ROOT, VAL_RAT_IDS, network_runs, regression_mse


COLORS = {"CNN": "#287D8E", "SNN": "#BA6073", "linear": "#B48536"}
INK = "#263743"


def configure_style(font_path=None):
    """Load the requested font explicitly, including Windows user fonts."""
    if font_path is None:
        candidates = [
            Path(os.environ.get("LOCALAPPDATA", ""))
            / "Microsoft/Windows/Fonts/HelveticaNeueLight.otf",
            Path(os.environ.get("WINDIR", "C:/Windows"))
            / "Fonts/HelveticaNeueLight.otf",
        ]
        font_path = next((path for path in candidates if path.is_file()), None)
    if font_path is not None:
        font_manager.fontManager.addfont(str(font_path))
        family = font_manager.FontProperties(fname=font_path).get_name()
    else:
        family = "Helvetica Neue"
        font_manager.findfont(
            font_manager.FontProperties(family=family, weight="light"),
            fallback_to_default=False,
        )
    plt.rcParams.update({
        "font.family": family,
        "font.weight": 300,
        "axes.labelweight": 300,
        "axes.titleweight": 300,
        "font.size": 15,
        "axes.titlesize": 21,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.edgecolor": "#CBD3D8",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.facecolor": "#FAFBFC",
        "figure.facecolor": "white",
        "grid.color": "#DFE5E9",
        "grid.linewidth": 0.7,
        "legend.frameon": False,
        "legend.fontsize": 13,
        "svg.fonttype": "path",
        "pdf.fonttype": 42,
        "savefig.dpi": 240,
    })
    return str(font_path) if font_path else family


def summarize_histories(runs, key, epochs):
    histories = [run[key] for run in runs.values()]
    for history in histories:
        if len(history) != epochs or not all(
            isinstance(value, (int, float)) and math.isfinite(value) and value >= 0
            for value in history
        ):
            raise ValueError(f"Invalid {key} for a {epochs}-epoch run")
    return [
        {"epoch": epoch, "mean": mean(values), "stdev": stdev(values), "runs": len(values)}
        for epoch, values in enumerate(zip(*histories, strict=True), start=1)
    ]


def draw_panel(axis, cohorts, history_key, baseline, title, colors=None):
    colors = COLORS if colors is None else colors
    for name, cohort in cohorts.items():
        stats = cohort[history_key]
        epochs = [row["epoch"] for row in stats]
        means = [row["mean"] for row in stats]
        # Squared error cannot be negative; truncate the displayed band at zero.
        lower = [max(0, row["mean"] - row["stdev"]) for row in stats]
        upper = [row["mean"] + row["stdev"] for row in stats]
        axis.fill_between(epochs, lower, upper, color=colors[name], alpha=0.17, linewidth=0)
        axis.plot(epochs, means, color=colors[name], linewidth=2.2,
                  linestyle=cohort.get("linestyle", "-"),
                  label=f"{name}  ·  {cohort['count']} runs")
    if baseline is not None:
        axis.axhline(baseline, color=COLORS["linear"], linewidth=1.7,
                     linestyle=(0, (5, 3)), label=f"Linear regression  ·  {baseline:.6f}")
    axis.set(title=title, xlabel="Epoch", ylabel="Mean squared error (fat fraction²)")
    axis.set_xlim(1, max(cohort["epochs"] for cohort in cohorts.values()))
    axis.set_ylim(bottom=0)
    axis.grid(axis="y")
    axis.set_axisbelow(True)
    axis.tick_params(length=0, pad=8)
    axis.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
    axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    axis.legend(loc="upper right")


def save_figure(figure, output_dir, stem):
    for extension in ("png", "pdf", "svg"):
        path = output_dir / f"{stem}.{extension}"
        figure.savefig(path, bbox_inches="tight", facecolor="white")
        print(path)
    plt.close(figure)


def learning_rate_comparison(cohorts, output_dir):
    """Compare validation learning-rate curves: SNN left, CNN right."""
    rates = sorted({rate for groups in cohorts.values() for rate in groups})
    palette = ["#6778AC", "#287D8E", "#BA6073", "#6C946D", "#8566A1"]
    if len(rates) > len(palette):
        palette = [plt.get_cmap("viridis")(i / (len(rates) - 1)) for i in range(len(rates))]
    rate_colors = dict(zip(rates, palette))
    figure, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    figure.subplots_adjust(left=0.07, right=0.80, bottom=0.12, top=0.84, wspace=0.27)
    figure.text(0.07, 0.94, "Validation MSE across learning rates", fontsize=26)
    for axis, name, title in zip(
        axes, ("SNN", "CNN"), ("(a) SNN", "(b) CNN"), strict=True
    ):
        groups, colors = {}, {}
        for rate, group in sorted(cohorts[name].items()):
            label = f"LR {rate:g}"
            groups[label] = {**group, "linestyle": ":" if name == "SNN" else "-"}
            colors[label] = rate_colors[rate]
        draw_panel(axis, groups, "val_history", None, title, colors=colors)
        axis.get_legend().remove()
    # Shared colors identify the same learning rate in both panels.
    handles = [Line2D([], [], color=rate_colors[rate], linewidth=2.2,
                      label=f"LR {rate:g}") for rate in rates]
    figure.legend(handles=handles, loc="center left", bbox_to_anchor=(0.82, 0.48),
                  handlelength=3, labelspacing=1.1)
    save_figure(figure, output_dir, "learning_rate_comparison")


def spike_steps_comparison(results_dir, output_dir):
    """Plot total spikes, comparing steps only within otherwise identical setups."""
    series = {}
    for config, records in load_groups(results_dir / "benchmarking_spikes", "spikes"):
        if config["model"] != "SNN":
            continue
        params = dict(config["init_params"])
        steps = params.pop("steps")
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
            raise ValueError(f"Invalid SNN step count: {steps}")
        setup = {**config, "init_params": params}
        key = json.dumps(setup, sort_keys=True)
        totals = metric_values(records, "total_spikes")
        if len(totals) < 2:
            raise ValueError(f"Need at least two spike files for steps={steps} to calculate stdev")
        series.setdefault(key, []).append({
            "steps": steps, "mean": mean(totals), "stdev": stdev(totals),
            "files": len(totals), "source_files": [path.name for path, _ in records],
        })
    if not series:
        raise ValueError("No SNN spike benchmarks found")
    figure, axis = plt.subplots(figsize=(10.8, 6.5))
    figure.subplots_adjust(left=0.11, right=0.97, bottom=0.12, top=0.88)
    figure.text(0.11, 0.94, "Total spikes across SNN steps", fontsize=23)
    summaries = []
    ticks = set()
    palette = [COLORS["SNN"], COLORS["CNN"], "#8566A1", "#6C946D"]
    for index, (key, rows) in enumerate(sorted(series.items())):
        config = json.loads(key)
        rows.sort(key=lambda row: row["steps"])
        steps = [row["steps"] for row in rows]
        ticks.update(steps)
        means = [row["mean"] / 1e6 for row in rows]
        lower = [max(0, row["mean"] - row["stdev"]) / 1e6 for row in rows]
        upper = [(row["mean"] + row["stdev"]) / 1e6 for row in rows]
        color = palette[index % len(palette)]
        label = "Mean ± 1 sample standard deviation"
        if len(series) > 1:
            label = (f"{config['device_name']}, threshold={config['init_params'].get('threshold')}, "
                     f"frames={config.get('frames_evaluated')}, lr={config.get('learning_rate')}")
        axis.fill_between(steps, lower, upper, color=color, alpha=0.17, linewidth=0)
        axis.plot(steps, means, color=color, linestyle=":", linewidth=2.4, marker="o",
                  markersize=7, label=label)
        if len(series) == 1:
            for step, value in zip(steps, means, strict=True):
                axis.annotate(f"{value:.2f} M", (step, value), xytext=(0, 12),
                              textcoords="offset points", ha="center", color=INK)
        summaries.append({"configuration": config, "statistics": rows})
    frame_counts = {item["configuration"].get("frames_evaluated") for item in summaries}
    ylabel = "Total spikes (millions)"
    if len(frame_counts) == 1 and None not in frame_counts:
        ylabel = f"Total spikes over {next(iter(frame_counts))} frames (millions)"
    axis.set(xlabel="SNN simulation steps", ylabel=ylabel)
    axis.set_xticks(sorted(ticks))
    axis.set_ylim(bottom=0)
    axis.margins(x=0.07, y=0.15)
    axis.grid(axis="y")
    axis.set_axisbelow(True)
    axis.tick_params(length=0, pad=8)
    axis.legend(loc="upper left")
    save_figure(figure, output_dir, "spikes_vs_steps")
    (output_dir / "spikes_vs_steps.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
    )
    return summaries


def spike_report_figures(summaries, output_dir):
    """Plot per-frame activity, layer firing rates, and final validation tradeoff."""
    series = {}
    for summary in summaries:
        config = summary["configuration"]
        params = dict(config["init_params"])
        steps = params.pop("steps")
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
            raise ValueError(f"Invalid SNN step count: {steps}")
        key = json.dumps({**config, "init_params": params}, sort_keys=True)
        series.setdefault(key, []).append(summary)
    for index, (key, groups) in enumerate(sorted(series.items()), start=1):
        groups.sort(key=lambda group: group["configuration"]["init_params"]["steps"])
        steps = [group["configuration"]["init_params"]["steps"] for group in groups]
        suffix = "" if len(series) == 1 else f"_{index}"

        def values(metric):
            means = [group["metrics"][metric]["mean"] for group in groups]
            deviations = [group["metrics"][metric]["stdev"] for group in groups]
            if any(value is None for value in deviations):
                raise ValueError("Spike report figures require at least two files per configuration")
            return means, deviations

        def new_plot(title):
            figure, axis = plt.subplots(figsize=(11.5, 7))
            figure.subplots_adjust(left=0.15, right=0.96, bottom=0.14, top=0.87)
            figure.text(0.15, 0.94, title, fontsize=24)
            axis.grid(axis="y")
            axis.set_axisbelow(True)
            axis.tick_params(length=0, pad=9)
            return figure, axis

        figure, axis = new_plot("Spike activity per frame")
        means, deviations = values("spikes_per_frame")
        means = [value / 1000 for value in means]
        deviations = [value / 1000 for value in deviations]
        axis.fill_between(steps, [max(0, m - s) for m, s in zip(means, deviations)],
                          [m + s for m, s in zip(means, deviations)],
                          color=COLORS["SNN"], alpha=0.17, linewidth=0)
        axis.plot(steps, means, ":o", color=COLORS["SNN"], linewidth=2.5, markersize=8,
                  label="Mean ± 1 standard deviation")
        axis.set(xlabel="SNN simulation steps", ylabel="Spikes per frame (thousands)")
        axis.set_xticks(steps)
        axis.set_ylim(bottom=0)
        axis.margins(x=0.06, y=0.12)
        axis.legend(loc="upper left")
        save_figure(figure, output_dir, f"spikes_per_frame_vs_steps{suffix}")

        figure, axis = new_plot("Firing rates by LIF layer")
        for layer, color in zip((1, 2, 3), (COLORS["CNN"], COLORS["SNN"], "#8566A1")):
            means, deviations = values(f"lif{layer}_rate_pct")
            axis.fill_between(steps, [max(0, m - s) for m, s in zip(means, deviations)],
                              [min(100, m + s) for m, s in zip(means, deviations)],
                              color=color, alpha=0.13, linewidth=0)
            axis.plot(steps, means, "-o", color=color, linewidth=2.5, markersize=7,
                      label=f"LIF {layer}")
        axis.set(xlabel="SNN simulation steps", ylabel="Firing rate (%)")
        axis.set_xticks(steps)
        axis.set_ylim(bottom=0)
        axis.margins(x=0.06, y=0.15)
        axis.legend(loc="lower right")
        save_figure(figure, output_dir, f"layer_firing_rates_vs_steps{suffix}")

        figure, axis = new_plot("Validation error versus spike activity")
        spike_means, spike_deviations = values("spikes_per_frame")
        mse_means, mse_deviations = values("final_val_mse")
        colors = ["#6778AC", "#287D8E", "#BA6073", "#6C946D", "#8566A1"]
        for i, step in enumerate(steps):
            axis.errorbar(spike_means[i] / 1000, mse_means[i],
                          xerr=spike_deviations[i] / 1000, yerr=mse_deviations[i],
                          fmt="o", markersize=9, capsize=5, elinewidth=1.8,
                          color=colors[i % len(colors)], label=f"{step} step{'s' if step != 1 else ''}")
        axis.set(xlabel="Spikes per frame (thousands)", ylabel="Final validation MSE (logged units)")
        axis.set_xlim(left=0)
        axis.margins(x=0.09, y=0.18)
        axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=False)
        axis.legend(loc="upper right", title="Simulation steps")
        save_figure(figure, output_dir, f"validation_mse_vs_spikes_per_frame{suffix}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--targets", type=Path,
                        default=ROOT / "Training_Data/training_targets.csv")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "artifact")
    parser.add_argument("--font-path", type=Path, help="Path to Helvetica Neue Light OTF/TTF")
    parser.add_argument("--cnn-epochs", type=int, default=25)
    parser.add_argument("--snn-epochs", type=int, default=75)
    args = parser.parse_args()
    font = configure_style(args.font_path)
    baseline = regression_mse(args.targets) / 10000
    cohorts = {}
    learning_rate_cohorts = {}
    for name, spiking, epochs in (
        ("CNN", False, args.cnn_epochs), ("SNN", True, args.snn_epochs)
    ):
        runs = network_runs(args.results_dir, spiking, epochs)
        cohorts[name] = {
            "count": len(runs), "epochs": epochs, "source_files": list(runs),
            **{key: summarize_histories(runs, key, epochs)
               for key in ("train_history", "val_history")},
        }
        grouped_runs = {}
        for filename, run in runs.items():
            rate = run.get("learning_rate")
            if not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
                raise ValueError(f"Invalid learning rate in {filename}")
            grouped_runs.setdefault(rate, {})[filename] = run
        learning_rate_cohorts[name] = {}
        for rate, group in sorted(grouped_runs.items()):
            if len(group) < 2:
                raise ValueError(f"Need at least two {name} runs at learning rate {rate} for stdev")
            learning_rate_cohorts[name][rate] = {
                "count": len(group), "epochs": epochs, "source_files": list(group),
                **{key: summarize_histories(group, key, epochs)
                   for key in ("train_history", "val_history")},
            }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for key, title, stem in (
        ("train_history", "Training MSE", "training_mse"),
        ("val_history", "Validation MSE", "validation_mse"),
    ):
        figure, axis = plt.subplots(figsize=(10.8, 6.5))
        figure.subplots_adjust(left=0.11, right=0.97, bottom=0.12, top=0.88)
        figure.text(0.11, 0.94, f"CNN & SNN  /  {title}", fontsize=23)
        draw_panel(axis, cohorts, key, baseline, "")
        save_figure(figure, args.output_dir, stem)

    figure, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    figure.subplots_adjust(left=0.07, right=0.98, bottom=0.12, top=0.84, wspace=0.25)
    figure.text(0.07, 0.94, "Learning to estimate liver fat", fontsize=26)
    for axis, key, title in zip(axes, ("train_history", "val_history"),
                                ("(a) Training", "(b) Validation"), strict=True):
        draw_panel(axis, cohorts, key, baseline, title)
    save_figure(figure, args.output_dir, "mse_comparison")
    learning_rate_comparison(learning_rate_cohorts, args.output_dir)
    spike_summaries = spike_steps_comparison(args.results_dir, args.output_dir)
    spike_report_figures(summarize_spikes(args.results_dir), args.output_dir)

    # Preserve exact plotted statistics and cohort membership for reproducibility.
    with (args.output_dir / "mse_curves.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "split", "epoch", "mean", "stdev", "runs"])
        writer.writeheader()
        for name, cohort in cohorts.items():
            for key, split in (("train_history", "training"), ("val_history", "validation")):
                writer.writerows({"model": name, "split": split, **row} for row in cohort[key])
    with (args.output_dir / "learning_rate_curves.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "model", "learning_rate", "split", "epoch", "mean", "stdev", "runs"
        ])
        writer.writeheader()
        for name, groups in learning_rate_cohorts.items():
            for rate, group in groups.items():
                for key, split in (("train_history", "training"), ("val_history", "validation")):
                    writer.writerows({"model": name, "learning_rate": rate, "split": split, **row}
                                     for row in group[key])
    metadata = {
        "spike_benchmarks": spike_summaries,
        "font": font, "validation_rat_ids": VAL_RAT_IDS,
        "linear_regression_fraction_mse": baseline,
        "network_target_scale": "Assumed fat fraction; not recorded in logs",
        "band": "mean +/- sample standard deviation (ddof=1), lower bound clipped at zero",
        "cohorts": {name: {key: value for key, value in cohort.items()
                           if key not in ("train_history", "val_history")}
                    for name, cohort in cohorts.items()},
        "learning_rate_cohorts": {
            name: {str(rate): {key: value for key, value in group.items()
                              if key not in ("train_history", "val_history")}
                   for rate, group in groups.items()}
            for name, groups in learning_rate_cohorts.items()
        },
    }
    (args.output_dir / "graph_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Font: {font}")


if __name__ == "__main__":
    main()
