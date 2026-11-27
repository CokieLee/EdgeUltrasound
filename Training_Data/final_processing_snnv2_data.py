#!/usr/bin/env python3
r"""Export paired selected RF and normalized FP32 tokens for the SNN V2.

Run with the config.json that belongs to the SNN V2 weights you will use:
  python final_processing_snnv2_data.py --model-config "c:/Users/ethan/Documents/Duke/Classes/Neuromorphic/EdgeUltrasound/src/fpga_CMODS7/SNN_V2/weights/config.json" 

Alternatively supply the EXACT exported model gain:
  python final_processing_snnv2_data.py --gain YOUR_MODEL_GAIN

Output files have matching names such as rat01_frame00.npy:
  unprocessed/rat01_frame00.npy      little-endian int16 [8,64,256]
  tokens/float32/rat01_frame00.npy   little-endian float32 [16,64]

The tokens ALREADY include gain division and clipping to [0,5]. Feed them
directly to an interface expecting normalized model input. Do not run the
partner's predict_tokens() wrapper on them: that wrapper divides by gain again.
For a PyTorch reference, pass them straight to model(tokens[None,...]).

Load .npy files with np.load(path, allow_pickle=False). They include a NumPy
header, so do not treat the entire .npy file as a raw UART payload.

Optional: --save-amplitude-summaries saves pre-gain FP32 values separately in
amplitude_summaries/float32/. --rats 1 --frames 0 exports one quick test pair.
Rerunning an identical export verifies and reuses completed pairs. A changed
gain, source file, or selection requires a NEW output directory.

Self-test: python export_snnv2_data.py --self-test
"""
import argparse
import csv
import hashlib
import json
import math
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


# The Windows defaults requested for this dataset.
DEFAULT_INPUT_DIR = Path(
    r"C:\Users\ethan\Documents\Duke\Classes\Neuromorphic\Stanford_Murine_RawRFData_TrainingFormatted"
)
DEFAULT_OUTPUT_DIR = Path(
    r"C:\Users\ethan\Documents\Duke\Classes\Neuromorphic\Stanford_Murine_Processed_RFData_SNNV2"
)
# Optional: set this to the matching SNN V2 config.json to omit the CLI flag.
DEFAULT_MODEL_CONFIG = None

EVENT_IDS = np.array([0, 54, 109, 164, 218, 273, 328, 383], dtype=np.int64)
CHANNEL_IDS = np.arange(0, 128, 2, dtype=np.int64)
SAMPLE_IDS = np.arange(0, 1000, 4, dtype=np.int64)
SOURCE_SHAPE = (10, 384, 128, 1000)
CLIP = 5.0
VERSION = "snnv2_paired_rf_tokens_v1"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def atomic_npy(path, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("wb") as f:
            np.save(f, array, allow_pickle=False)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def resolve_gain(args):
    config_path = args.model_config
    if config_path is None and args.gain is None:
        config_path = DEFAULT_MODEL_CONFIG
    if config_path is not None and args.gain is not None:
        raise ValueError("Choose --model-config or --gain, not both")
    provenance = {"model_config": None, "model_config_sha256": None}
    if config_path is not None:
        config_path = Path(config_path).expanduser().resolve()
        saved = json.loads(config_path.read_text(encoding="utf-8-sig"))
        config = saved.get("config", saved)
        if "gain" not in config:
            raise ValueError(f"No top-level gain in {config_path}; use the SNN V2 model config")
        gain = float(config["gain"])
        for key in ("clip", "input_clip"):
            if key in config and float(config[key]) != CLIP:
                raise ValueError(f"This export requires clipping to [0,5], but {key}={config[key]}")
        provenance = {"model_config": str(config_path), "model_config_sha256": sha256_file(config_path)}
    elif args.gain is not None:
        gain = args.gain
    else:
        raise ValueError("Supply --model-config pointing to the matching SNN V2 config.json, "
                         "or --gain with its exact saved gain. Gain is never fitted from these files.")
    if not math.isfinite(gain) or gain <= 0 or gain > np.finfo(np.float32).max:
        raise ValueError("Gain must be positive, finite, and representable in float32")
    with np.errstate(under="ignore"):
        effective = np.float32(gain)
    if effective <= 0:
        raise ValueError("Gain underflows float32")
    return float(effective), {**provenance, "gain_as_supplied": gain, "effective_float32_gain": float(effective)}


def select_padded_rf(rf, frame_index):
    """Read eight small HDF5 blocks; never materialize a full [384,128,1000] frame."""
    selected = np.stack([
        np.asarray(rf[frame_index, int(event), 0:128:2, 0:1000:4])
        for event in EVENT_IDS
    ], axis=0)
    if selected.shape != (8, 64, 250):
        raise ValueError(f"Unexpected selected RF shape {selected.shape}")
    # Preserve signed amplitudes. Padding belongs at the END of the sample axis.
    selected = np.ascontiguousarray(selected, dtype="<i2")
    padded = np.pad(selected, ((0, 0), (0, 0), (0, 6)), mode="constant")
    return np.ascontiguousarray(padded, dtype="<i2")


def amplitude_summaries(padded_rf):
    """Exact source-model feature ordering: sample window, event slot, channel group.

    Cast BEFORE abs: abs(int16(-32768)) would overflow. Each block sums 128
    absolute int16 values, so its maximum sum is 4,194,304. That is safe in
    int32 and exactly representable in float32. Division by 128 is also exact;
    this produces the same summaries as the partner's FP32 abs().mean((2,4)).
    """
    if padded_rf.shape != (8, 64, 256) or padded_rf.dtype.kind != "i" or padded_rf.dtype.itemsize != 2:
        raise ValueError("Expected signed int16 padded RF with shape [8,64,256]")
    if np.any(padded_rf[:, :, 250:] != 0):
        raise ValueError("The six padded sample positions must be zero")
    blocks = padded_rf.astype(np.int32).reshape(8, 8, 8, 16, 16)
    sums = np.abs(blocks).sum(axis=(2, 4), dtype=np.int32)
    means = sums.astype(np.float32) / np.float32(128)
    return np.ascontiguousarray(means.transpose(2, 0, 1).reshape(16, 64), dtype="<f4")


def normalized_tokens(summaries, gain):
    if summaries.shape != (16, 64) or not np.isfinite(summaries).all() or np.any(summaries < 0):
        raise ValueError("Expected finite nonnegative amplitude summaries with shape [16,64]")
    # Use float32 division, matching the exported model's float32 input path.
    with np.errstate(over="ignore"):
        normalized = summaries.astype(np.float32) / np.float32(gain)
    return np.ascontiguousarray(np.clip(normalized, np.float32(0), np.float32(CLIP)), dtype="<f4")


def read_optional_targets(source_dir):
    path = source_dir / "training_targets.csv"
    if not path.is_file():
        return {}, None
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if rows and not {"rat_index", "liver_fat_percent"}.issubset(rows[0]):
        return {}, {"path": str(path), "note": "Target columns unavailable; arrays do not require labels"}
    targets = {int(r["rat_index"]): float(r["liver_fat_percent"]) for r in rows}
    if len(targets) != len(rows) or not all(math.isfinite(v) for v in targets.values()):
        raise ValueError("Duplicate rat IDs or nonfinite target values in training_targets.csv")
    return targets, {"path": str(path), "sha256": sha256_file(path)}


def discover_sources(source_dir, rats, h5py, dataset_key):
    found = {}
    for path in source_dir.iterdir():
        match = re.fullmatch(r"Rat(\d+)_FSA_frames\.(?:h5|hdf5)", path.name, flags=re.IGNORECASE)
        if not path.is_file() or match is None:
            continue
        rat = int(match.group(1))
        if rats is not None and rat not in rats:
            continue
        if rat in found:
            raise ValueError(f"Two matching source files for rat {rat}")
        found[rat] = path
    if rats is not None and set(rats) != set(found):
        raise FileNotFoundError(f"Missing source files for rat IDs {sorted(set(rats)-set(found))}")
    if not found:
        raise FileNotFoundError(f"No Rat*_FSA_frames.h5/.hdf5 files found in {source_dir}")
    sources = []
    for rat, path in sorted(found.items()):
        with h5py.File(path, "r") as f:
            if dataset_key not in f:
                raise KeyError(f"Dataset {dataset_key!r} missing in {path}")
            rf = f[dataset_key]
            if tuple(rf.shape) != SOURCE_SHAPE:
                raise ValueError(f"{path.name}: expected {SOURCE_SHAPE}, got {rf.shape}")
            if rf.dtype.kind != "i" or rf.dtype.itemsize != 2:
                raise ValueError(f"{path.name}: expected original signed int16 RF, got {rf.dtype}; "
                                 "do not use a normalized float cache")
            dtype = rf.dtype.str
        stat = path.stat()
        sources.append({"rat_id": rat, "path": str(path.resolve()), "size_bytes": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns, "shape": list(SOURCE_SHAPE), "dtype": dtype})
    return sources


def verified_pair(root, entry, include_summaries):
    if entry is None:
        return False
    required = {"unprocessed", "tokens_float32"}
    if include_summaries:
        required.add("amplitude_summaries_float32")
    if set(entry.get("files", {})) != required:
        return False
    for kind, item in entry["files"].items():
        path = root / item["path"]
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            return False
        array = np.load(path, allow_pickle=False)
        shape, dtype = ((8, 64, 256), np.dtype("<i2")) if kind == "unprocessed" else ((16, 64), np.dtype("<f4"))
        if array.shape != shape or array.dtype != dtype or not array.flags.c_contiguous:
            return False
    return True


def export_dataset(args):
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("Install dependencies with: python -m pip install numpy h5py") from exc
    gain, gain_source = resolve_gain(args)
    source_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {source_dir}")
    if source_dir == output_dir:
        raise ValueError("Input and output directories must be different")
    if args.rats is not None and (len(set(args.rats)) != len(args.rats) or min(args.rats) < 1):
        raise ValueError("Rat IDs must be distinct positive integers")
    frames = list(range(10)) if args.frames is None else sorted(args.frames)
    if not frames or len(set(frames)) != len(frames) or min(frames) < 0 or max(frames) > 9:
        raise ValueError("Frame indices must be distinct integers from 0 through 9")
    sources = discover_sources(source_dir, args.rats, h5py, args.dataset_key)
    targets, targets_info = read_optional_targets(source_dir)
    protocol = {
        "version": VERSION, "dataset_key": args.dataset_key,
        "source_shape": list(SOURCE_SHAPE), "source_files": sources, "frame_indices": frames,
        "event_indices_zero_based": EVENT_IDS.tolist(),
        "recieve_channel_indices_zero_based": CHANNEL_IDS.tolist(),
        "sample_indices_zero_based": SAMPLE_IDS.tolist(), "sample_end_padding": 6,
        "unprocessed": {"directory": "unprocessed", "shape": [8, 64, 256],
                        "dtype": "<i2", "normalization": "none; original signed amplitudes"},
        "amplitude_summary": {"recieve_channels_per_group": 8, "samples_per_window": 16,
                              "operation": "mean absolute amplitude over 128 values, including zero padding",
                              "shape": [16, 64], "save_separately": args.save_amplitude_summaries},
        "tokens_float32": {"directory": "tokens/float32", "shape": [16, 64], "dtype": "<f4",
                           "gain": gain, "clip": [0.0, CLIP], "gain_division_applied": True,
                           "clipping_applied": True, "per_frame_max_normalization": False,
                           "order": "C: token/sample window first, then event slot, then recieve channel group",
                           "feature_index": "8 * event_slot + recieve_channel_group",
                           "flatten_index": "64 * token_index + feature_index"},
        "gain_source": gain_source, "targets": targets_info,
        "file_format": "NumPy .npy with dtype/shape header; little-endian; C-contiguous",
    }
    identity = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    expected_names = {f"rat{s['rat_id']:02d}_frame{frame:02d}.npy" for s in sources for frame in frames}
    subdirs = ["unprocessed", "tokens/float32"]
    if args.save_amplitude_summaries:
        subdirs.append("amplitude_summaries/float32")
    for subdir in subdirs:
        path = output_dir / subdir
        path.mkdir(parents=True, exist_ok=True)
        unexpected = {p.name for p in path.glob("*.npy")} - expected_names
        if unexpected:
            raise ValueError(f"Unexpected arrays in {path}; use a fresh output directory: {sorted(unexpected)[:5]}")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("export_identity") != identity:
            raise ValueError("Existing manifest uses a different gain/config/source/selection. "
                             "Use a NEW --output-dir to avoid mixing incompatible tokens.")
    else:
        if any(list((output_dir / subdir).glob("*.npy")) for subdir in subdirs):
            raise ValueError("Arrays exist without this exporter's manifest; use a fresh output directory")
        manifest = {"export_identity": identity, "protocol": protocol, "created_at": utc_now(), "frames": []}
    records = {r["name"]: r for r in manifest["frames"]}
    manifest.update(status="running", updated_at=utc_now())
    manifest.pop("error", None)
    atomic_json(manifest_path, manifest)
    written = reused = 0
    print(f"Input:  {source_dir}\nOutput: {output_dir}\nGain:   {gain:.12g}; clip [0,5]")
    print(f"Selected {len(sources)} rat files x {len(frames)} frames. Frame IDs in filenames are zero-based.")
    try:
        for source in sources:
            rat = source["rat_id"]
            print(f"Rat {rat:02d}: {Path(source['path']).name}", flush=True)
            with h5py.File(source["path"], "r") as h:
                rf = h[args.dataset_key]
                for frame in frames:
                    name = f"rat{rat:02d}_frame{frame:02d}.npy"
                    if verified_pair(output_dir, records.get(name), args.save_amplitude_summaries):
                        reused += 1
                        continue
                    padded = select_padded_rf(rf, frame)
                    summaries = amplitude_summaries(padded)
                    tokens = normalized_tokens(summaries, gain)
                    with np.errstate(over="ignore"):
                        clipped_count = int(np.count_nonzero(summaries / np.float32(gain) > np.float32(CLIP)))
                    arrays = {"unprocessed": ("unprocessed", padded),
                              "tokens_float32": ("tokens/float32", tokens)}
                    if args.save_amplitude_summaries:
                        arrays["amplitude_summaries_float32"] = ("amplitude_summaries/float32", summaries)
                    files = {}
                    for kind, (subdir, array) in arrays.items():
                        path = output_dir / subdir / name
                        atomic_npy(path, array)
                        files[kind] = {"path": path.relative_to(output_dir).as_posix(),
                                       "shape": list(array.shape), "dtype": array.dtype.str,
                                       "sha256": sha256_file(path)}
                    records[name] = {
                        "name": name, "rat_id": rat, "frame_index_zero_based": frame,
                        "source_file": source["path"], "target_fat_percent": targets.get(rat),
                        "files": files, "token_min": float(tokens.min()), "token_max": float(tokens.max()),
                        "amplitude_summary_max": float(summaries.max()),
                        "token_count": int(tokens.size),
                        "clipped_token_count": clipped_count,
                    }
                    written += 1
                    manifest["frames"] = sorted(records.values(), key=lambda r: (r["rat_id"], r["frame_index_zero_based"]))
                    manifest["updated_at"] = utc_now()
                    atomic_json(manifest_path, manifest)
            stat = Path(source["path"]).stat()
            if (stat.st_size, stat.st_mtime_ns) != (source["size_bytes"], source["mtime_ns"]):
                raise RuntimeError(f"Source changed during export: {source['path']}")
        manifest.update(status="completed", completed_at=utc_now(), frame_count=len(records))
    except BaseException as exc:
        manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                        error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        manifest["updated_at"] = utc_now()
        atomic_json(manifest_path, manifest)
    print(f"Complete: {len(records)} matched pairs ({written} written; {reused} verified and reused).")
    print("tokens/float32 files are ALREADY normalized. Do not divide by gain again on the host/device.")
    print(f"Metadata: {manifest_path}")
    return manifest


def self_test():
    """Independent indexing/mean/padding tests; optional real HDF5 integration."""
    rng = np.random.default_rng(42)
    selected = rng.integers(-32768, 32768, size=(8, 64, 250), dtype=np.int16)
    selected[0, 0, 0] = -32768
    padded = np.pad(selected, ((0, 0), (0, 0), (0, 6)))
    actual = amplitude_summaries(padded)
    expected = np.zeros((16, 64), dtype=np.float32)
    # Independent, explicit definition; does not repeat the reshape reduction.
    for t in range(16):
        for e in range(8):
            for group in range(8):
                block = selected[e, group * 8:(group + 1) * 8, t * 16:min((t + 1) * 16, 250)]
                expected[t, e * 8 + group] = sum(abs(int(v)) for v in block.reshape(-1)) / 128
    np.testing.assert_array_equal(actual, expected)
    const = np.full((8, 64, 256), -32768, dtype=np.int16)
    const[:, :, 250:] = 0
    raw = amplitude_summaries(const)
    np.testing.assert_array_equal(raw[:15], np.full((15, 64), 32768, dtype=np.float32))
    np.testing.assert_array_equal(raw[15], np.full(64, 20480, dtype=np.float32))
    normalized = normalized_tokens(raw, 8192)
    assert normalized.dtype == np.dtype('<f4') and normalized.flags.c_contiguous
    np.testing.assert_array_equal(normalized[:15], np.full((15, 64), 4, dtype=np.float32))
    np.testing.assert_array_equal(normalized[15], np.full(64, 2.5, dtype=np.float32))
    np.testing.assert_array_equal(normalized_tokens(raw, 1), np.full((16, 64), 5, dtype=np.float32))
    class IndexedRF:
        def __init__(self): self.calls = []
        def __getitem__(self, indices):
            frame, event, channels, samples = indices
            self.calls.append((frame, event, channels, samples))
            c = np.arange(128)[channels]
            s = np.arange(1000)[samples]
            return (event * 20 + c[:, None] * 3 + s[None, :] - 1000).astype(np.int16)
    fixture = IndexedRF()
    selected_padded = select_padded_rf(fixture, 3)
    assert len(fixture.calls) == 8 and selected_padded.shape == (8, 64, 256)
    np.testing.assert_array_equal(selected_padded[:, :, 250:], 0)
    assert selected_padded[2, 7, 15] == EVENT_IDS[2] * 20 + 14 * 3 + 60 - 1000
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'test.npy'
        atomic_npy(path, selected_padded)
        np.testing.assert_array_equal(np.load(path, allow_pickle=False), selected_padded)
        assert np.load(path, allow_pickle=False).dtype == np.dtype('<i2')
    print('PASS: original-index selection, signed int16 minimum, independent 128-value means,')
    print('token ordering, final-window padding, float32 gain/clipping, and .npy roundtrip.')
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        print('Optional PyTorch reference check skipped: PyTorch is not installed or required.')
    else:
        reference = F.pad(torch.from_numpy(selected.astype(np.float32)), (0, 6))
        reference = reference.reshape(8, 8, 8, 16, 16).abs().mean(dim=(2, 4)).permute(2, 0, 1).reshape(16, 64)
        np.testing.assert_array_equal(actual, reference.numpy())
        for test_gain in (8192.0, 12345.678):
            wanted = (reference / test_gain).clamp(0, 5).numpy()
            np.testing.assert_allclose(normalized_tokens(actual, test_gain), wanted, rtol=1e-6, atol=1e-7)
        print('PASS: summaries and normalized tokens agree with the partner\'s PyTorch operations.')
    try:
        import h5py
    except ImportError:
        print('HDF5 integration skipped: h5py is not installed. Install h5py and rerun --self-test.')
        return
    import contextlib
    import io
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        config = root / 'config.json'
        config.write_text(json.dumps({'gain': 8192, 'clip': 5}))
        with h5py.File(root / 'Rat1_FSA_frames.h5', 'w') as h:
            rf = h.create_dataset('rf_data', SOURCE_SHAPE, dtype='<i2',
                                  chunks=(1, 1, 16, 250), fillvalue=0)
            for e, event in enumerate(EVENT_IDS):
                rf[0, int(event), ::2, ::4] = selected[e]
        args = parser().parse_args(['--input-dir', str(root), '--output-dir', str(root / 'output'),
                                    '--model-config', str(config), '--rats', '1', '--frames', '0',
                                    '--save-amplitude-summaries'])
        with contextlib.redirect_stdout(io.StringIO()):
            report = export_dataset(args)
            again = export_dataset(args)
        assert report['status'] == 'completed' and len(again['frames']) == 1
        output = root / 'output'
        np.testing.assert_array_equal(np.load(output / 'unprocessed/rat01_frame00.npy'), padded)
        np.testing.assert_array_equal(np.load(output / 'amplitude_summaries/float32/rat01_frame00.npy'), expected)
        np.testing.assert_array_equal(np.load(output / 'tokens/float32/rat01_frame00.npy'), normalized_tokens(expected, 8192))
        assert verified_pair(output, again['frames'][0], True)
        changed = parser().parse_args(['--input-dir', str(root), '--output-dir', str(output),
                                      '--gain', '1024', '--rats', '1', '--frames', '0'])
        try:
            export_dataset(changed)
        except ValueError:
            pass
        else:
            raise AssertionError('Changed normalization must not mix with an existing export')
    print('PASS: real HDF5 export, matched filenames, manifest, normalized tokens, resume, and gain-change protection.')


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input-dir', type=Path, default=DEFAULT_INPUT_DIR)
    p.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    gain = p.add_mutually_exclusive_group()
    gain.add_argument('--model-config', type=Path, help='SNN V2 config.json containing the matching positive gain')
    gain.add_argument('--gain', type=float, help='Exact gain belonging to the model weights; never refitted here')
    p.add_argument('--dataset-key', default='rf_data')
    p.add_argument('--rats', nargs='+', type=int, help='Optional rat IDs; default all matching input files')
    p.add_argument('--frames', nargs='+', type=int, help='Optional zero-based frame indices; default 0..9')
    p.add_argument('--save-amplitude-summaries', action='store_true', help='Also save pre-gain values separately')
    p.add_argument('--self-test', action='store_true')
    return p


def main():
    p = parser()
    args = p.parse_args()
    if args.self_test:
        self_test()
        return
    # Validate normalization before touching output files (and before importing h5py).
    try:
        resolve_gain(args)
        export_dataset(args)
    except (ValueError, FileNotFoundError, KeyError, RuntimeError) as exc:
        p.exit(1, f'ERROR: {exc}\n')


if __name__ == '__main__':
    main()
