"""Format the first FSA acquisition for each available murine rat.

The output RF tensor has shape:
    [frames, receive_events, channels, samples]

Run from the workspace root or with absolute paths:
    python EdgeUltrasound/Training_Data/training_rawRF_formatting.py

Optional paths:
    python EdgeUltrasound/Training_Data/training_rawRF_formatting.py \
        --input-root Stanford_Murine_RawRFData \
        --output-dir Stanford_Murine_RawRFData_TrainingFormatted
"""

import argparse
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = SCRIPT_ROOT / "Stanford_Murine_RawRFData"
DEFAULT_OUTPUT_DIR = SCRIPT_ROOT / "Stanford_Murine_RawRFData_TrainingFormatted"
DEFAULT_PLOT_DIR = SCRIPT_ROOT / "Stanford_Murine_RawRFData_Photos"

# Set to 1 to skip valid outputs that already exist; set to 0 to regenerate all rats.
SKIP_EXISTING_RATS = 0
INITIAL_SAMPLES_TO_REMOVE = 64
RANDOM_TRACE_COUNT = 16
TOTAL_SAMPLES_TO_KEEP = 1000
# Set to None to process all discovered rats, or use a list such as [1, 4].
RATS_TO_PROCESS = None

if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from FSA_view import (  # noqa: E402
    all_datasets,
    candidate_datasets,
    decimation_factor,
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


def trim_frame_samples(frame_data):
    required_sample_count = INITIAL_SAMPLES_TO_REMOVE + TOTAL_SAMPLES_TO_KEEP
    if frame_data.shape[-1] < required_sample_count:
        raise ValueError(
            f"Frame sample count {frame_data.shape[-1]} is less than the "
            f"required {required_sample_count} samples"
        )
    return frame_data[
        ..., INITIAL_SAMPLES_TO_REMOVE:required_sample_count
    ]


def is_valid_formatted_output(output_path):
    if not output_path.exists():
        return False
    try:
        with h5py.File(output_path, mode="r") as archive:
            required_files = {
                "rf_data",
                "receive_indices",
                "start_samples",
                "end_samples",
                "frame_indices",
            }
            if not required_files.issubset(archive.keys()):
                return False
            return archive["rf_data"].ndim == 4
    except (OSError, KeyError):
        return False


def plot_random_receive_traces(
    rf_data,
    rat_name,
    plot_path,
    random_generator=None,
):
    frame_count, event_count, channel_count, sample_count = rf_data.shape
    available_trace_count = frame_count * event_count
    trace_count = min(RANDOM_TRACE_COUNT, available_trace_count)
    if trace_count == 0 or channel_count == 0 or sample_count == 0:
        print("  plot skipped: no RF traces available")
        return

    random_generator = random_generator or np.random.default_rng()
    trace_indices = random_generator.choice(
        available_trace_count,
        size=trace_count,
        replace=False,
    )
    sample_indices = np.arange(sample_count)

    figure, axes = plt.subplots(4, 4, figsize=(16, 12), sharex=True)
    axes = axes.ravel()
    for subplot_index, trace_index in enumerate(trace_indices):
        frame_index, event_index = divmod(int(trace_index), event_count)
        channel_index = int(random_generator.integers(channel_count))
        axis = axes[subplot_index]
        axis.plot(
            sample_indices,
            rf_data[frame_index, event_index, channel_index],
            linewidth=0.8,
        )
        axis.set_title(
            f"frame {frame_index}, event {event_index}, channel {channel_index}",
            fontsize=8,
        )
        axis.grid(True, alpha=0.25)

    figure.suptitle(f"Random RF traces: {rat_name}")
    figure.supxlabel("Sample index")
    figure.supylabel("Raw RF amplitude")
    figure.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)
    print(f"  trace plot: {plot_path}")


def save_formatted_rat(
    output_path,
    rf_data,
    frame_events_by_frame,
    sampling_frequency,
    decimation_factor_value,
    initial_samples_removed,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output_path = output_path.with_suffix(".tmp.h5")
    event_count = rf_data.shape[1]
    channel_count = rf_data.shape[2]
    sample_count = rf_data.shape[3]
    frame_count = rf_data.shape[0]
    receive_indices = np.array(
        [[event["receive_index"] for event in frame_events] for frame_events in frame_events_by_frame],
        dtype=np.int64,
    )
    start_samples = np.array(
        [[event["start_sample"] for event in frame_events] for frame_events in frame_events_by_frame],
        dtype=np.int64,
    )
    end_samples = np.array(
        [[event["end_sample"] for event in frame_events] for frame_events in frame_events_by_frame],
        dtype=np.int64,
    )
    frame_indices = np.array(
        [[event["frame_index"] for event in frame_events] for frame_events in frame_events_by_frame],
        dtype=np.int64,
    )
    start_depths = np.array(
        [[event["start_depth"] for event in frame_events] for frame_events in frame_events_by_frame],
        dtype=np.float64,
    )
    end_depths = np.array(
        [[event["end_depth"] for event in frame_events] for frame_events in frame_events_by_frame],
        dtype=np.float64,
    )

    try:
        with h5py.File(temporary_output_path, mode="w") as archive:
            dataset = archive.create_dataset(
                "rf_data",
                shape=rf_data.shape,
                dtype=rf_data.dtype,
                chunks=(1, 1, channel_count, sample_count),
                compression="lzf",
            )
            for frame_index in range(frame_count):
                for event_index in range(event_count):
                    dataset[frame_index, event_index] = rf_data[frame_index, event_index]

            archive.create_dataset("receive_indices", data=receive_indices)
            archive.create_dataset("start_samples", data=start_samples)
            archive.create_dataset("end_samples", data=end_samples)
            archive.create_dataset("frame_indices", data=frame_indices)
            archive.create_dataset("start_depths", data=start_depths)
            archive.create_dataset("end_depths", data=end_depths)
            archive.create_dataset("frame_count", data=np.array(frame_count, dtype=np.int64))
            archive.create_dataset("event_count", data=np.array(event_count, dtype=np.int64))
            archive.create_dataset("channel_count", data=np.array(channel_count, dtype=np.int64))
            archive.create_dataset("sample_count", data=np.array(sample_count, dtype=np.int64))
            archive.create_dataset(
                "initial_samples_removed",
                data=np.array(initial_samples_removed, dtype=np.int64),
            )
            archive.create_dataset(
                "sampling_frequency_hz",
                data=np.array(sampling_frequency if sampling_frequency is not None else np.nan),
            )
            archive.create_dataset(
                "decim_factor",
                data=np.array(
                    decimation_factor_value
                    if decimation_factor_value is not None
                    else np.nan
                ),
            )
            archive.attrs["format"] = "Verasonics FSA training RF"
            archive.attrs["tensor_order"] = "frames,events,channels,samples"
        os.replace(temporary_output_path, output_path)
    except Exception:
        if temporary_output_path.exists():
            temporary_output_path.unlink()
        raise


def format_rat(rat_dir, output_dir):
    output_path = output_dir / f"{rat_dir.name}_FSA_frames.h5"
    legacy_output_path = output_dir / f"{rat_dir.name}_FSA_frames.npz"
    if SKIP_EXISTING_RATS:
        if is_valid_formatted_output(output_path):
            print(f"{rat_dir.name}: existing valid HDF5 output found; skipped")
            return None
        if legacy_output_path.exists():
            print(f"{rat_dir.name}: existing NPZ output found; skipped")
            return None

    source_file = first_fsa_file(rat_dir)
    if source_file is None:
        print(f"{rat_dir.name}: no FSA file found; skipped")
        return None

    print(f"\n{rat_dir.name}")
    print(f"  source: {source_file.relative_to(rat_dir)}")

    with h5py.File(source_file, "r") as h5_file:
        rf_dataset = dataset_for_file(h5_file)
        receive_events = receive_event_descriptors(h5_file)
        if not receive_events:
            raise RuntimeError("No valid Receive events found.")
        sampling_frequency = sampling_frequency_hz(h5_file)
        receive_decimation_factor = decimation_factor(h5_file)
        depth_mismatches = [
            event
            for event in receive_events
            if not np.isclose(event["start_depth"], 2.0)
        ]
        if depth_mismatches:
            print("  startDepth values other than 2.0:")
            for event in depth_mismatches:
                print(
                    f"    Receive index {event['receive_index']}: "
                    f"{event['start_depth']}"
                )
        frame_indices = sorted({event["frame_index"] for event in receive_events})
        first_frame_data, first_frame_events = extract_frame(
            rf_dataset,
            receive_events,
            frame_index=frame_indices[0],
        )
        first_frame_data = trim_frame_samples(first_frame_data)
        frame_shape = first_frame_data.shape
        full_shape = (len(frame_indices),) + frame_shape
        frame_events_by_frame = [first_frame_events]

        with tempfile.TemporaryDirectory(prefix=f"{rat_dir.name}_rf_") as temporary_dir:
            temporary_path = Path(temporary_dir) / "rf_data.npy"
            rf_data = np.lib.format.open_memmap(
                temporary_path,
                mode="w+",
                dtype=first_frame_data.dtype,
                shape=full_shape,
            )
            rf_data[0] = first_frame_data
            for output_frame_index, frame_index in enumerate(frame_indices[1:], start=1):
                frame_data, current_events = extract_frame(
                    rf_dataset,
                    receive_events,
                    frame_index=frame_index,
                )
                frame_data = trim_frame_samples(frame_data)
                if current_events and frame_data.shape != frame_shape:
                    raise ValueError(
                        f"Frame {frame_index} shape {frame_data.shape} differs from {frame_shape}"
                    )
                if len(current_events) != len(first_frame_events):
                    raise ValueError(
                        f"Frame {frame_index} has {len(current_events)} events; "
                        f"expected {len(first_frame_events)} for this file"
                    )
                rf_data[output_frame_index] = frame_data
                frame_events_by_frame.append(current_events)
            rf_data.flush()
            plot_random_receive_traces(
                rf_data,
                rat_dir.name,
                DEFAULT_PLOT_DIR / f"{rat_dir.name}_random_traces.png",
            )
            save_formatted_rat(
                output_path,
                rf_data,
                frame_events_by_frame,
                sampling_frequency,
                receive_decimation_factor,
                INITIAL_SAMPLES_TO_REMOVE,
            )
            del rf_data

    print(f"  frames: {full_shape[0]}")
    print(f"  events per frame: {full_shape[1]}")
    print(f"  channels per event: {full_shape[2]}")
    print(f"  samples per channel: {full_shape[3]}")
    print(f"  initial samples removed per channel: {INITIAL_SAMPLES_TO_REMOVE}")
    print(f"  sampling frequency (Hz): {sampling_frequency}")
    print(f"  output: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    input_root = args.input_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rat_dirs = sorted(path for path in input_root.glob("Rat*") if path.is_dir())
    if RATS_TO_PROCESS is not None:
        requested_rats = set(RATS_TO_PROCESS)
        rat_dirs = [path for path in rat_dirs if rat_number(path) in requested_rats]
    if not rat_dirs:
        selection = "selected" if RATS_TO_PROCESS is not None else "Rat*"
        raise FileNotFoundError(f"No {selection} directories found under {input_root}")

    print(f"Input root: {input_root}")
    print(f"Output directory: {output_dir}")
    print(f"Rats selected: {len(rat_dirs)}")
    outputs = []
    for rat_dir in rat_dirs:
        outputs.append(format_rat(rat_dir, output_dir))
    print(f"\nFormatted rat files: {sum(output is not None for output in outputs)}")


if __name__ == "__main__":
    main()
