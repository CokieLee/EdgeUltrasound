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
import shutil
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import h5py
import numpy as np


SCRIPT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = SCRIPT_ROOT / "Stanford_Murine_RawRFData"
DEFAULT_OUTPUT_DIR = SCRIPT_ROOT / "Stanford_Murine_RawRFData_TrainingFormatted"

# Set to 1 to skip valid outputs that already exist; set to 0 to regenerate all rats.
SKIP_EXISTING_RATS = 1
INITIAL_SAMPLES_TO_REMOVE = 64

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


def dataset_for_file(h5_file):
    datasets = all_datasets(h5_file)
    candidates = candidate_datasets(datasets)
    if not candidates:
        raise RuntimeError("No numeric RF-like dataset was found.")
    return candidates[0]


def write_array_to_zip(zip_file, name, array):
    with zip_file.open(name, "w", force_zip64=True) as destination:
        np.lib.format.write_array(destination, np.asarray(array), allow_pickle=False)


def is_valid_formatted_output(output_path):
    if not output_path.exists():
        return False
    try:
        with zipfile.ZipFile(output_path, mode="r") as archive:
            required_files = {
                "rf_data.npy",
                "frame_count.npy",
                "event_count.npy",
                "channel_count.npy",
                "sample_count.npy",
                "initial_samples_removed.npy",
            }
            if not required_files.issubset(archive.namelist()):
                return False
            return archive.testzip() is None
    except (OSError, zipfile.BadZipFile):
        return False


def save_formatted_rat(
    output_path,
    rf_data,
    frame_events_by_frame,
    sampling_frequency,
    initial_samples_removed,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
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

    with zipfile.ZipFile(
        output_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=1,
    ) as archive:
        write_array_to_zip(archive, "rf_data.npy", rf_data)
        write_array_to_zip(archive, "receive_indices.npy", receive_indices)
        write_array_to_zip(archive, "start_samples.npy", start_samples)
        write_array_to_zip(archive, "end_samples.npy", end_samples)
        write_array_to_zip(archive, "frame_indices.npy", frame_indices)
        write_array_to_zip(archive, "frame_count.npy", np.array(frame_count, dtype=np.int64))
        write_array_to_zip(archive, "event_count.npy", np.array(event_count, dtype=np.int64))
        write_array_to_zip(archive, "channel_count.npy", np.array(channel_count, dtype=np.int64))
        write_array_to_zip(archive, "sample_count.npy", np.array(sample_count, dtype=np.int64))
        write_array_to_zip(
            archive,
            "initial_samples_removed.npy",
            np.array(initial_samples_removed, dtype=np.int64),
        )
        write_array_to_zip(
            archive,
            "sampling_frequency_hz.npy",
            np.array(sampling_frequency if sampling_frequency is not None else np.nan),
        )


def format_rat(rat_dir, output_dir):
    output_path = output_dir / f"{rat_dir.name}_FSA_frames.npz"
    if SKIP_EXISTING_RATS and is_valid_formatted_output(output_path):
        print(f"{rat_dir.name}: existing valid output found; skipped")
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
        frame_indices = sorted({event["frame_index"] for event in receive_events})
        first_frame_data, first_frame_events = extract_frame(
            rf_dataset,
            receive_events,
            frame_index=frame_indices[0],
        )
        if first_frame_data.shape[-1] <= INITIAL_SAMPLES_TO_REMOVE:
            raise ValueError(
                f"Frame sample count {first_frame_data.shape[-1]} is not larger than "
                f"INITIAL_SAMPLES_TO_REMOVE={INITIAL_SAMPLES_TO_REMOVE}"
            )
        first_frame_data = first_frame_data[..., INITIAL_SAMPLES_TO_REMOVE:]
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
                frame_data = frame_data[..., INITIAL_SAMPLES_TO_REMOVE:]
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
            save_formatted_rat(
                output_path,
                rf_data,
                frame_events_by_frame,
                sampling_frequency,
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
    if not rat_dirs:
        raise FileNotFoundError(f"No Rat* directories found under {input_root}")

    print(f"Input root: {input_root}")
    print(f"Output directory: {output_dir}")
    print(f"Rats discovered: {len(rat_dirs)}")
    outputs = []
    for rat_dir in rat_dirs:
        outputs.append(format_rat(rat_dir, output_dir))
    print(f"\nFormatted rat files: {sum(output is not None for output in outputs)}")


if __name__ == "__main__":
    main()
