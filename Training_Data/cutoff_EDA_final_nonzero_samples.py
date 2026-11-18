"""Inspect RF amplitudes at one sample for each Receive-event/channel trace.

The script reads the first FSA acquisition for each selected rat, collects the
RF amplitude at one selected sample for every channel in every Receive event,
and displays a histogram for each rat. No data or plots are written to disk.
"""

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = SCRIPT_ROOT / "Stanford_Murine_RawRFData"

# Set to None to process all discovered rats, or use a list such as [1, 4].
RATS_TO_PROCESS = [2]
TARGET_SAMPLE_NUMBER = 1064

if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from FSA_view import (  # noqa: E402
    all_datasets,
    candidate_datasets,
    extract_frame,
    receive_event_descriptors,
    sampling_frequency_hz,
)


def first_fsa_file(rat_dir):
    fsa_files = sorted(rat_dir.rglob("DATA_FSA_*.mat"))
    return fsa_files[0] if fsa_files else None


def rat_number(rat_dir):
    try:
        return int(rat_dir.name[3:])
    except ValueError:
        return None


def dataset_for_file(h5_file):
    datasets = all_datasets(h5_file)
    candidates = candidate_datasets(datasets)
    if not candidates:
        raise RuntimeError("No numeric RF-like dataset was found.")
    return candidates[0]


def collect_sample_values(rf_dataset, receive_events, sample_number):
    """Return amplitudes at a 1-based sample number for all event/channel traces."""
    sample_index = sample_number - 1
    frame_indices = sorted({event["frame_index"] for event in receive_events})
    sample_values = []

    for frame_index in frame_indices:
        frame_data, _ = extract_frame(
            rf_dataset,
            receive_events,
            frame_index=frame_index,
        )
        for event_data in frame_data:
            if sample_index < 0 or sample_index >= event_data.shape[0]:
                print(
                    f"  sample {sample_number} is outside event range "
                    f"1-{event_data.shape[0]}; skipping value collection"
                )
                return np.asarray([])
            sample_values.extend(event_data[sample_index])

    return np.asarray(sample_values)


def show_first_receive_trace(
    rat_name,
    rf_dataset,
    receive_events,
    sampling_frequency,
):
    """Display the first Receive event and first channel before target lookup."""
    if sampling_frequency is None or sampling_frequency <= 0:
        print(f"  first-trace plot skipped: invalid sampling frequency {sampling_frequency}")
        return

    first_frame_index = min(event["frame_index"] for event in receive_events)
    frame_data, frame_events = extract_frame(
        rf_dataset,
        receive_events,
        frame_index=first_frame_index,
    )
    first_trace = frame_data[0, 0]
    time_us = np.arange(first_trace.size) / sampling_frequency * 1e6

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(time_us, first_trace)
    axis.set_xlabel("Time (microseconds)")
    axis.set_ylabel("Raw RF amplitude")
    axis.set_title(
        f"{rat_name}: first Receive event, first channel "
        f"({first_trace.size} samples)"
    )
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    plt.show(block=True)
    plt.close(figure)
    print(
        f"  first event/channel samples: {first_trace.size} "
        f"(Receive index {frame_events[0]['receive_index']})"
    )


def show_sample_value_histogram(
    rat_name,
    sample_values,
    sample_number,
):
    if sample_values.size == 0:
        print(f"  {rat_name}: no sample values found")
        return

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.hist(sample_values, bins=50, edgecolor="black")
    axis.set_xlabel("RF amplitude")
    axis.set_ylabel("Trace count")
    axis.set_title(f"{rat_name}: RF values at sample {sample_number}")
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    plt.show(block=True)
    plt.close(figure)


def inspect_rat(rat_dir):
    source_file = first_fsa_file(rat_dir)
    if source_file is None:
        print(f"{rat_dir.name}: no FSA file found; skipped")
        return

    print(f"\n{rat_dir.name}")
    print(f"  source: {source_file.relative_to(rat_dir)}")

    with h5py.File(source_file, "r") as h5_file:
        rf_dataset = dataset_for_file(h5_file)
        receive_events = receive_event_descriptors(h5_file)
        if not receive_events:
            raise RuntimeError("No valid Receive events found.")
        sampling_frequency = sampling_frequency_hz(h5_file)
        show_first_receive_trace(
            rat_dir.name,
            rf_dataset,
            receive_events,
            sampling_frequency,
        )
        sample_values = collect_sample_values(
            rf_dataset,
            receive_events,
            TARGET_SAMPLE_NUMBER,
        )

    print(f"  sampling frequency (Hz): {sampling_frequency}")
    print(f"  Receive events: {len(receive_events)}")
    print(f"  sample number: {TARGET_SAMPLE_NUMBER}")
    print(f"  values collected: {sample_values.size}")
    if sample_values.size:
        print(
            "  amplitude range: "
            f"{sample_values.min()} to {sample_values.max()}"
        )
    show_sample_value_histogram(
        rat_dir.name,
        sample_values,
        TARGET_SAMPLE_NUMBER,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    args = parser.parse_args()

    input_root = args.input_root.resolve()
    rat_dirs = sorted(path for path in input_root.glob("Rat*") if path.is_dir())
    if RATS_TO_PROCESS is not None:
        requested_rats = set(RATS_TO_PROCESS)
        rat_dirs = [path for path in rat_dirs if rat_number(path) in requested_rats]
    if not rat_dirs:
        selection = "selected" if RATS_TO_PROCESS is not None else "Rat*"
        raise FileNotFoundError(f"No {selection} directories found under {input_root}")

    print(f"Input root: {input_root}")
    print(f"Rats selected: {len(rat_dirs)}")
    for rat_dir in rat_dirs:
        inspect_rat(rat_dir)


if __name__ == "__main__":
    main()
