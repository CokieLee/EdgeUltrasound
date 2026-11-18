"""Summarize and visualize one Verasonics FSA MATLAB v7.3 acquisition.

Usage:
    py FSA_view.py
    py FSA_view.py path\to\DATA_FSA_....mat

The MAT files are HDF5 containers.
"""

import argparse
from collections import Counter
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_FILE = (
    Path(__file__).parent
    / "Stanford_Murine_RawRFData"
    / "Rat5"
    / "DATA_FSA_20190401_093235.mat"
)
DEFAULT_MASK_INITIAL_MICROSECONDS = 2.0
DEFAULT_CHANNEL_START = 0
DEFAULT_CHANNEL_COUNT = 16


def all_datasets(h5_file):
    datasets = []

    def collect(_name, value):
        if isinstance(value, h5py.Dataset):
            datasets.append(value)

    h5_file.visititems(collect)
    return datasets


def is_numeric(dataset):
    return np.issubdtype(dataset.dtype, np.number)


def dereference(h5_file, value):
    if isinstance(value, h5py.Reference):
        return h5_file[value]
    return value


def first_value(h5_file, value):
    value = dereference(h5_file, value)
    if isinstance(value, h5py.Dataset):
        if h5py.check_dtype(ref=value.dtype) is not None:
            reference = np.asarray(value[()]).reshape(-1)[0]
            return first_value(h5_file, reference)
        scalar = np.asarray(value[()]).squeeze()
        return scalar.item() if scalar.size == 1 else scalar
    return value


def read_struct_field(h5_file, variable_name, field_name, index=0):
    """Resolve a scalar field from a MATLAB v7.3 struct or struct array."""
    variable = h5_file[variable_name]
    if isinstance(variable, h5py.Group) and field_name in variable:
        field = variable[field_name]
        if isinstance(field, h5py.Dataset) and field.size > 1:
            if h5py.check_dtype(ref=field.dtype) is not None:
                reference = np.asarray(field[()]).reshape(-1)[index]
                return first_value(h5_file, reference)
            value = np.asarray(field[()]).reshape(-1)[index]
            return value.item() if np.asarray(value).size == 1 else value
        return first_value(h5_file, field)
    if isinstance(variable, h5py.Dataset) and h5py.check_dtype(ref=variable.dtype) is not None:
        reference = np.asarray(variable[()]).reshape(-1)[index]
        variable = dereference(h5_file, reference)
    if isinstance(variable, h5py.Group) and field_name in variable:
        return first_value(h5_file, variable[field_name])
    return None


def sampling_frequency_hz(h5_file):
    decimated_sample_rate_mhz = read_struct_field(
        h5_file, "Receive", "decimSampleRate"
    )
    if decimated_sample_rate_mhz is None:
        return None
    return decimated_sample_rate_mhz * 1e6


def decimation_factor(h5_file):
    return read_struct_field(h5_file, "Receive", "decimFactor")


def receive_event_descriptors(h5_file):
    """Return one sample-range descriptor for every Verasonics Receive entry."""
    receive_variable = h5_file["Receive"]
    if not isinstance(receive_variable, h5py.Group):
        raise RuntimeError("Receive is not stored as a MATLAB struct group.")
    event_count = receive_variable["startSample"].size
    events = []
    for event_index in range(event_count):
        start_sample = read_struct_field(h5_file, "Receive", "startSample", event_index)
        end_sample = read_struct_field(h5_file, "Receive", "endSample", event_index)
        start_depth = read_struct_field(h5_file, "Receive", "startDepth", event_index)
        end_depth = read_struct_field(h5_file, "Receive", "endDepth", event_index)
        frame_number = read_struct_field(h5_file, "Receive", "framenum", event_index)
        if start_sample is None or end_sample is None:
            continue
        events.append(
            {
                "receive_index": event_index,
                "start_sample": int(start_sample),
                "end_sample": int(end_sample),
                "start_depth": float(start_depth) if start_depth is not None else np.nan,
                "end_depth": float(end_depth) if end_depth is not None else np.nan,
                "frame_index": max(0, int(frame_number) - 1) if frame_number is not None else 0,
            }
        )
    return events


def read_receive_event(rf_dataset, event, max_channels=None):
    """Read one Receive event as [samples, channels] using MATLAB 1-based indices."""
    start = event["start_sample"] - 1
    end = event["end_sample"]
    if rf_dataset.ndim != 3 or temporal_axis(rf_dataset) != 2:
        raise ValueError(f"Expected RF data shaped [frames, channels, samples], got {rf_dataset.shape}")
    if start < 0 or end > rf_dataset.shape[2]:
        raise ValueError(f"Receive event range {start + 1}:{end} is outside RF data")
    frame_index = event["frame_index"]
    if frame_index >= rf_dataset.shape[0]:
        raise ValueError(f"Receive frame {frame_index} is outside RF data")
    return np.asarray(rf_dataset[frame_index, :max_channels, start:end]).T


def extract_frame(rf_dataset, receive_events, frame_index=0, max_channels=None):
    """Extract one frame as [receive events, channels, samples]."""
    frame_events = [
        event for event in receive_events if event["frame_index"] == frame_index
    ]
    if not frame_events:
        raise ValueError(f"No receive events found for frame {frame_index}")

    event_arrays = [
        read_receive_event(rf_dataset, event, max_channels=max_channels)
        for event in frame_events
    ]
    event_lengths = {event_array.shape[0] for event_array in event_arrays}
    if len(event_lengths) != 1:
        raise ValueError(f"Receive events have different sample lengths: {sorted(event_lengths)}")

    frame_data = np.stack(event_arrays, axis=0).transpose(0, 2, 1)
    return frame_data, frame_events


def save_frame(output_path, frame_data, frame_events, sampling_frequency):
    np.savez_compressed(
        output_path,
        rf_frame=frame_data,
        receive_indices=np.array([event["receive_index"] for event in frame_events]),
        start_samples=np.array([event["start_sample"] for event in frame_events]),
        end_samples=np.array([event["end_sample"] for event in frame_events]),
        frame_indices=np.array([event["frame_index"] for event in frame_events]),
        sampling_frequency_hz=sampling_frequency,
    )
    print(f"Saved frame matrix to: {output_path}")
    print(f"Saved rf_frame shape: {frame_data.shape} [events, channels, samples]")


def candidate_datasets(datasets):
    candidates = [
        dataset
        for dataset in datasets
        if is_numeric(dataset) and dataset.ndim >= 2 and dataset.size > 1024
    ]

    def score(dataset):
        path = dataset.name.lower()
        name_bonus = 10_000 if "rcvdata" in path else 0
        return name_bonus + dataset.size

    return sorted(candidates, key=score, reverse=True)


def temporal_axis(dataset):
    """Return the longest axis, which is the RF sample axis in these files."""
    return int(np.argmax(dataset.shape))


def print_summary(file_path, h5_file, datasets, rf_dataset, sibling_files, receive_events):
    sequence = file_path.name.split("_")[1] if "_" in file_path.name else "unknown"
    samples_per_wave = read_struct_field(h5_file, "Receive", "samplesPerWave")
    center_frequency_mhz = read_struct_field(h5_file, "Trans", "frequency")
    sampling_frequency = sampling_frequency_hz(h5_file)
    if sampling_frequency is not None:
        sampling_frequency_mhz = sampling_frequency / 1e6
        sample_period_ns = 1_000 / sampling_frequency_mhz
    else:
        sampling_frequency_mhz = None
        sample_period_ns = None
    print(f"File: {file_path}")
    print(f"Size: {file_path.stat().st_size / 1024**3:.2f} GiB")
    print(f"Sequence: {sequence}")
    print(f"MAT/HDF5 top-level variables: {list(h5_file.keys())}")
    print(f"HDF5 datasets found: {len(datasets)}")
    print(f"Selected RF-like dataset: {rf_dataset.name}")
    print(f"RF dtype: {rf_dataset.dtype}, shape: {rf_dataset.shape}")
    print(f"RF storage size: {rf_dataset.size * rf_dataset.dtype.itemsize / 1024**2:.1f} MiB")
    print(f"Transducer center frequency: {center_frequency_mhz} MHz")
    print(f"Samples per wave: {samples_per_wave}")
    print(f"RF sampling frequency: {sampling_frequency_mhz} MHz")
    print(f"RF sample period: {sample_period_ns} ns")
    print(f"Receive events with valid sample ranges: {len(receive_events)}")
    frame_indices = [event["frame_index"] for event in receive_events]
    print(f"RF frames represented by Receive events: {len(set(frame_indices))}")

    counts = Counter(path.name.split("_")[1] for path in sibling_files if "_" in path.name)
    print(f"Rat folder acquisitions: {len(sibling_files)} total")
    print("Acquisitions by sequence: " + ", ".join(f"{key}={value}" for key, value in sorted(counts.items())))
    print("These files share the rat-level label; they are repeated acquisitions, not independent rats.")


def plot_rf(
    file_path,
    frame_data,
    sampling_frequency,
    frame_events,
    event_index=0,
    mask_initial_microseconds=DEFAULT_MASK_INITIAL_MICROSECONDS,
    channel_start=DEFAULT_CHANNEL_START,
    channel_count=DEFAULT_CHANNEL_COUNT,
):
    if sampling_frequency is None:
        raise RuntimeError("Cannot convert samples to time without RF sampling metadata.")
    event = frame_events[event_index]
    event_data = np.real(frame_data[event_index].T)
    mask_initial_samples = int(
        np.ceil(mask_initial_microseconds * 1e-6 * sampling_frequency)
    )
    if mask_initial_samples < 0 or mask_initial_samples > event_data.shape[0]:
        raise ValueError("The requested time mask exceeds the event sample range")
    if channel_start < 0 or channel_start >= event_data.shape[1]:
        raise ValueError("channel_start must identify an available receive channel")
    if channel_count < 1:
        raise ValueError("channel_count must be positive")

    masked_event_data = event_data.copy()
    masked_event_data[:mask_initial_samples, :] = 0
    selected_channels = range(
        channel_start,
        min(channel_start + channel_count, event_data.shape[1]),
    )
    selected_channels = list(selected_channels)
    if not selected_channels:
        raise ValueError("No receive channels selected for plotting")

    time_us = np.arange(event_data.shape[0]) / sampling_frequency * 1e6
    actual_mask_microseconds = mask_initial_samples / sampling_frequency * 1e6
    print(f"Selected Receive index: {event['receive_index']}")
    print(f"Selected event samples: {event['start_sample']}:{event['end_sample']}")
    print(f"Selected RF frame index: {event['frame_index']}")
    print(f"Selected event tensor shape: {event_data.shape} [samples, channels]")
    print(f"Mask: first {mask_initial_samples} samples ({actual_mask_microseconds:.6f} microseconds)")

    for plot_channel in selected_channels:
        figure, axes = plt.subplots(
            1,
            2,
            figsize=(14, 5),
            sharex=True,
            constrained_layout=True,
        )
        axes[0].plot(time_us, event_data[:, plot_channel], linewidth=0.7)
        axes[0].set_title(f"Original RF, channel {plot_channel}")
        axes[0].set_xlabel("Time from event start (microseconds)")
        axes[0].set_ylabel("Recorded amplitude")
        axes[0].grid(alpha=0.25)

        axes[1].plot(time_us, masked_event_data[:, plot_channel], linewidth=0.7)
        axes[1].set_title(f"Masked RF, first {actual_mask_microseconds:.2f} microseconds removed")
        axes[1].set_xlabel("Time from event start (microseconds)")
        axes[1].set_ylabel("Recorded amplitude")
        axes[1].grid(alpha=0.25)
        figure.suptitle(f"{file_path.name}, Receive event {event['receive_index']}")

        plt.show(block=True)
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", nargs="?", type=Path, default=DEFAULT_FILE)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument(
        "--mask-initial-microseconds",
        type=float,
        default=DEFAULT_MASK_INITIAL_MICROSECONDS,
    )
    parser.add_argument("--channel-start", type=int, default=DEFAULT_CHANNEL_START)
    parser.add_argument("--channel-count", type=int, default=DEFAULT_CHANNEL_COUNT)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "Rat5_FSA_frame0.npz",
    )
    args = parser.parse_args()

    file_path = args.file.resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"MAT file not found: {file_path}")

    sibling_files = sorted(file_path.parent.glob("DATA_*.mat"))
    with h5py.File(file_path, "r") as h5_file:
        datasets = all_datasets(h5_file)
        candidates = candidate_datasets(datasets)
        if not candidates:
            raise RuntimeError("No numeric RF-like dataset was found in the MAT file.")
        rf_dataset = candidates[0]
        sampling_frequency = sampling_frequency_hz(h5_file)
        receive_events = receive_event_descriptors(h5_file)
        if not receive_events:
            raise RuntimeError("No Receive entries with valid sample ranges were found.")
        print_summary(file_path, h5_file, datasets, rf_dataset, sibling_files, receive_events)
        frame_data, frame_events = extract_frame(
            rf_dataset,
            receive_events,
            frame_index=args.frame_index,
        )
        print(f"Selected frame: {args.frame_index}")
        print(f"Selected frame matrix shape: {frame_data.shape} [events, channels, samples]")
        save_frame(args.output.resolve(), frame_data, frame_events, sampling_frequency)
        plot_rf(
            file_path,
            frame_data,
            sampling_frequency,
            frame_events,
            mask_initial_microseconds=args.mask_initial_microseconds,
            channel_start=args.channel_start,
            channel_count=args.channel_count,
        )


if __name__ == "__main__":
    main()