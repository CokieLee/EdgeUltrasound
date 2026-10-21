"""Print benchmark statistics and export spike report tables and figures.

Run: python reports/midterm/benchmark_calculation.py
Timing samples are pooled within identical model/device/configuration groups.
MAC and spike summaries use one value per JSON file. Repeated spike evaluations
are not assumed to be independently trained models. Timing units are preserved
exactly as recorded in run_timings_ms; no unit conversion is inferred.
"""

import argparse
import csv
from collections import defaultdict
import json
import math
from pathlib import Path
import re
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[2]
SPIKE_METRICS = (
    "best_val_mse", "final_val_mse", "training_seconds", "spike_eval_seconds",
    "total_spikes", "spikes_per_frame", "spikes_per_second",
    "lif1_spikes", "lif2_spikes", "lif3_spikes",
    "lif1_rate_pct", "lif2_rate_pct", "lif3_rate_pct",
)


def load_groups(directory, section):
    """Keep hardware, architecture, and evaluation/training settings separate."""
    paths = sorted(directory.glob("*.json"))
    if not paths:
        raise ValueError(f"No benchmark JSON files found in {directory}")
    groups = defaultdict(list)
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            record = json.load(handle)
        params = record["init_params"]
        if not isinstance(params.get("spiking"), bool):
            raise ValueError(f"Missing/invalid spiking flag in {path}")
        experiment = record.get("notes", record.get("note", ""))
        if section == "spikes":
            # The run number identifies a repetition rather than a configuration.
            experiment = re.sub(r",\s*run\s+\d+\s*$", "", experiment)
        configuration = {
            "model": "SNN" if params["spiking"] else "CNN",
            "device": record["device"],
            "device_name": record["device_name"],
            "init_params": params,
            "experiment": experiment,
        }
        keys = ("sample_shape",) if section == "timing" else ("parameter_count",)
        if section == "spikes":
            keys += ("epochs", "learning_rate", "weight_decay", "train_batch_size",
                     "val_batch_size", "frames_evaluated")
        for key in keys:
            if key in record:
                configuration[key] = record[key]
        groups[json.dumps(configuration, sort_keys=True)].append((path, record))
    return [(json.loads(key), records) for key, records in sorted(groups.items())]


def metric_values(records, metric, pooled=False):
    values = []
    for path, record in records:
        value = record[metric]
        samples = value if pooled else [value]
        if not samples or not all(
            isinstance(item, (int, float)) and not isinstance(item, bool)
            and math.isfinite(item) and item >= 0 for item in samples
        ):
            raise ValueError(f"Invalid {metric} in {path}")
        values.extend(samples)
    return values


def print_metric(label, values):
    deviation = f"{stdev(values):.9g}" if len(values) > 1 else "N/A"
    print(f"  {label:<29} {len(values):>6} {mean(values):>17.9g} {deviation:>17}")


def summarize_spikes(results_dir):
    """Summarize each step/configuration without treating files as new seeds."""
    summaries = []
    for config, records in load_groups(results_dir / "benchmarking_spikes", "spikes"):
        if config["model"] != "SNN":
            continue
        metrics = {}
        for metric in SPIKE_METRICS:
            values = metric_values(records, metric)
            metrics[metric] = {"mean": mean(values),
                               "stdev": stdev(values) if len(values) > 1 else None,
                               "n": len(values)}
        summaries.append({"configuration": config, "metrics": metrics,
                          "source_files": [path.name for path, _ in records]})
    if not summaries:
        raise ValueError("No SNN spike benchmarks found")
    return summaries


def export_spike_report(summaries, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "standard_deviation": "Sample standard deviation across result files (ddof=1)",
        "repetitions": "Files may repeat the same trained model; not independent seeds",
        "firing_rates": "Logged lif1/2/3_rate_pct values, in percent",
        "validation_metric": "Final-epoch validation MSE in logged target units",
        "groups": summaries,
    }
    path = output_dir / "spike_report.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(path)
    path = output_dir / "spike_report.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["device", "device_name", "steps", "threshold", "learning_rate",
                  "epochs", "frames_evaluated", "metric", "mean", "stdev", "n"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            config = summary["configuration"]
            context = {key: config.get(key) for key in fields[:7]}
            context.update(steps=config["init_params"]["steps"],
                           threshold=config["init_params"].get("threshold"))
            for metric, stats in summary["metrics"].items():
                writer.writerow({**context, "metric": metric, **stats})
    print(path)


def print_section(results_dir, section):
    directory = {"timing": "benchmarking", "macs": "benchmarking_macs",
                 "spikes": "benchmarking_spikes"}[section]
    heading = {"timing": "Inference timing", "macs": "MACs and parameter counts",
               "spikes": "Spike evaluations"}[section]
    print(f"\n{heading}")
    for config, records in load_groups(results_dir / directory, section):
        label = f"{config['model']} | {config['device_name']} ({config['device']})"
        if section == "spikes":
            params = config["init_params"]
            label += (f" | steps={params.get('steps')}, threshold={params.get('threshold')}"
                      f" | lr={config.get('learning_rate')}, epochs={config.get('epochs')}"
                      f" | frames={config.get('frames_evaluated')}")
        if section == "timing":
            label += f" | shape={config.get('sample_shape')}"
        print(f"\n{label} | {len(records)} file(s)")
        print(f"  Experiment: {config['experiment']}; init_params={config['init_params']}")
        print(f"  {'Metric':<29} {'N':>6} {'Mean':>17} {'Sample stdev':>17}")
        if section == "timing":
            print_metric("Latency (logged ms)", metric_values(records, "run_timings_ms", pooled=True))
        elif section == "macs":
            for metric in ("macs", "parameter_count"):
                print_metric(metric, metric_values(records, metric))
        else:
            for metric in SPIKE_METRICS:
                print_metric(metric, metric_values(records, metric))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--section", choices=("all", "timing", "macs", "spikes"), default="all")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "artifact")
    parser.add_argument("--font-path", type=Path)
    parser.add_argument("--no-figures", action="store_true", help="Print/export statistics without Matplotlib")
    args = parser.parse_args()
    print("Benchmark means and sample standard deviations (ddof=1).")
    print("N = pooled timing repetitions, or result files for MAC/spike metrics.")
    print("N/A = fewer than two observations; spike files may repeat the same trained model.")
    sections = ("timing", "macs", "spikes") if args.section == "all" else (args.section,)
    for section in sections:
        print_section(args.results_dir, section)
    if "spikes" in sections:
        summaries = summarize_spikes(args.results_dir)
        export_spike_report(summaries, args.output_dir)
        if not args.no_figures:
            if __package__:
                from .graph_generation import configure_style, spike_report_figures
            else:
                from graph_generation import configure_style, spike_report_figures
            configure_style(args.font_path)
            spike_report_figures(summaries, args.output_dir)


if __name__ == "__main__":
    main()
