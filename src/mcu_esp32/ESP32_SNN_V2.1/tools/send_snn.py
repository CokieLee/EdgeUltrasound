#!/usr/bin/env python3
"""Send one frame or token tensor to the ESP32-S3 SNN over serial."""
# to use: $tokens = 'C:\Users\ethan\Documents\Duke\Classes\Neuromorphic\Stanford_Murine_Processed_RFData_SNNV2\skip8_V2.1_1127\tokens\float32'
# python tools/send_snn.py COM11 --dummy
# python tools/send_snn.py COM11 --normalized "$tokens\rat01_frame00.npy"

import argparse
import json
from pathlib import Path
import time

import numpy as np


TOKEN_SHAPE = (16, 64)
RF_SHAPE = (384, 64, 250)
EVENT_IDS = (0, 54, 109, 164, 218, 273, 328, 383)
GAIN = np.float32(81.0234375)  # Trained skip8 checkpoint and ESP32 normalization.


def encode_rf(frame):
    if frame.shape != RF_SHAPE:
        raise ValueError(f"Expected an original RF frame of shape {RF_SHAPE}")
    selected = np.asarray(frame[list(EVENT_IDS)], dtype=np.float32)
    padded = np.pad(selected, ((0, 0), (0, 0), (0, 6)))
    pooled = np.abs(padded.reshape(8, 8, 8, 16, 16)).mean(axis=(2, 4))
    return pooled.transpose(2, 0, 1).reshape(TOKEN_SHAPE)


def smoke_tokens():
    values = np.arange(np.prod(TOKEN_SHAPE), dtype=np.int32)
    return ((values * 17 % 97).astype(np.float32) * 0.5).reshape(TOKEN_SHAPE)


def predict(link, tokens):
    if tokens.shape != TOKEN_SHAPE:
        raise ValueError(f"Expected tokens of shape {TOKEN_SHAPE}, got {tokens.shape}")
    tokens = np.asarray(tokens, dtype="<f4", order="C")
    if not np.isfinite(tokens).all():
        raise ValueError("Tokens must be finite")
    payload = tokens.tobytes()
    if link.write(payload) != len(payload):
        raise RuntimeError("The token payload was only partially written")
    link.flush()

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        line = link.readline()
        if line.startswith(b"SNN_RESULT "):
            _, score, percentage, elapsed = line.split()
            if link.read(1) != b"\xaa":
                raise RuntimeError("Missing ESP32 acknowledgment byte")
            return float(score), float(percentage), int(elapsed)
    raise RuntimeError("No ESP32 result received")


def input_paths(parser, args):
    source = args.normalized or args.input
    if source is None:
        return [None]
    if args.normalized is not None and source.is_dir():
        paths = sorted(path for path in source.glob("*.npy") if path.is_file())
        if not paths:
            parser.error(f"{source} contains no .npy files")
        return paths
    if not source.is_file():
        parser.error(f"Input file not found: {source}")
    return [source]


def load_tokens(path, normalized):
    if path is None:
        return smoke_tokens()
    data = np.load(path, allow_pickle=False)
    if not isinstance(data, np.ndarray):
        data.close()
        raise ValueError("Expected a NumPy .npy array")
    if not normalized:
        return encode_rf(data) if data.shape == RF_SHAPE else data
    if data.shape != TOKEN_SHAPE or data.dtype.kind != "f" or data.dtype.itemsize != 4:
        raise ValueError(f"Expected float32 tokens of shape {TOKEN_SHAPE}")
    if not np.isfinite(data).all() or np.any((data < 0) | (data > 5)):
        raise ValueError("Normalized tokens must be finite and in [0,5]")
    # The firmware applies the gain and clipping to the float32 wire values.
    return data * GAIN


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("port", help="Serial port, for example COM11 or /dev/ttyACM0")
    parser.add_argument("input", nargs="?", type=Path,
                        help="Legacy RF or pre-gain token .npy")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--normalized", type=Path, help="Normalized float32 .npy or folder of those files")
    mode.add_argument("--dummy", action="store_true", help="Send the original smoke test (default)")
    parser.add_argument("--output", type=Path, help="Write scores and inference time to JSON")
    args = parser.parse_args()
    if args.input is not None and (args.normalized is not None or args.dummy):
        parser.error("Use either a positional input, --normalized, or --dummy")
    paths = input_paths(parser, args)
    batch = args.normalized is not None and args.normalized.is_dir()

    import serial

    reports = []
    with serial.Serial(args.port, 500000, timeout=0.5, write_timeout=5) as link:
        time.sleep(2)
        link.reset_input_buffer()
        for path in paths:
            if path is not None:
                print(f"Frame {path.name}", flush=True)
            try:
                tokens = load_tokens(path, normalized=args.normalized is not None)
                score, percentage, elapsed = predict(link, tokens)
            except (OSError, ValueError, RuntimeError) as exc:
                raise SystemExit(f'{path if path is not None else "Dummy frame"}: {exc}') from exc
            print(f"Standardized score {score:.7f}")
            print(f"Predicted liver fat {percentage:.7f}%")
            print(f"On-device inference {elapsed} µs")
            report = dict(score=score, predicted_target_percent=percentage, inference_us=elapsed)
            if path is not None:
                report["input"] = str(path)
            reports.append(report)

    if batch:
        print(f"Processed {len(reports)} frames.")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output = dict(frames=reports) if batch else reports[0]
        args.output.write_text(json.dumps(output, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
