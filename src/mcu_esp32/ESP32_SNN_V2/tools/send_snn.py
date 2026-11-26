#!/usr/bin/env python3
"""Send one frame or token tensor to the ESP32-S3 SNN over serial."""
import argparse
import json
import math
from pathlib import Path
import struct
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


def token_payload(tokens):
    if tokens.shape != TOKEN_SHAPE:
        raise ValueError(f"Expected tokens of shape {TOKEN_SHAPE}, got {tokens.shape}")
    tokens = np.asarray(tokens, dtype="<f4", order="C")
    if not np.isfinite(tokens).all():
        raise ValueError("Tokens must be finite")
    return tokens.tobytes()


def send(link, payload):
    if link.write(payload) != len(payload):
        raise RuntimeError("The token payload was only partially written")
    link.flush()


def predict(link, tokens):
    send(link, token_payload(tokens))
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        line = link.readline()
        if line.startswith(b"SNN_RESULT "):
            _, score, percentage, elapsed = line.split()
            if link.read(1) != b"\xaa":
                raise RuntimeError("Missing ESP32 acknowledgment byte")
            return float(score), float(percentage), int(elapsed)
    raise RuntimeError("No ESP32 result received")


def power_reply(link, tag, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        line = link.readline().split()
        if line and line[0] == b"SNN_POWER_ERROR":
            raise RuntimeError(b" ".join(line[1:]).decode("ascii", errors="replace"))
        if line and line[0] == tag:
            return line[1:]
    raise TimeoutError(f"No {tag.decode()} received. Check that the power benchmark firmware is loaded.")


def power_benchmark(link, tokens, args):
    duration_ms = args.power_benchmark * 1000
    send(link, b"\xff"*4 + struct.pack("<I", duration_ms) + token_payload(tokens))
    settle_ms, requested_ms = map(int, power_reply(link, b"SNN_POWER_START", 5))
    if requested_ms != duration_ms or not 0 <= settle_ms <= 10000:
        raise RuntimeError("Unexpected benchmark start response")
    print(f"Inference starts in {settle_ms / 1000:g} s and runs for "
          f"{requested_ms / 1000:g} s. Read average rail current during that window.", flush=True)
    count_raw, elapsed_raw, score_raw = power_reply(
        link, b"SNN_POWER_RESULT", (settle_ms + duration_ms)/1000 + 10)
    count, elapsed_us, score = int(count_raw), int(elapsed_raw), float(score_raw)
    if link.read(1) != b"\xaa":
        raise RuntimeError("Missing ESP32 acknowledgment byte")
    if count <= 0 or elapsed_us < duration_ms * 1000 or not math.isfinite(score):
        raise RuntimeError("Invalid benchmark result")

    mean_us = elapsed_us / count
    rate = 1e6 / mean_us
    report = dict(mode="power_benchmark", requested_seconds=args.power_benchmark,
                  inferences=count, elapsed_us=elapsed_us, mean_us_per_inference=mean_us,
                  inferences_per_second=rate, score=score, rail_voltage_v=args.voltage)
    print(f"Finished {count} inferences in {elapsed_us/1e6:.3f} s")
    print(f"Throughput {rate:.2f} inferences/s, {mean_us:.2f} µs per inference including loop overhead")
    print(f"Standardized score {score:.7f}")
    if args.current_ma is None:
        print("Use --current-ma with your measured mean current to calculate rail power and energy.")
        return report

    power_mw = args.voltage * args.current_ma
    energy_uj = power_mw * 1000 / rate
    report.update(current_ma=args.current_ma, rail_power_mw=power_mw, energy_per_inference_uj=energy_uj)
    print(f"Rail power {power_mw:.2f} mW; rail energy {energy_uj:.2f} µJ/inference")
    if args.idle_current_ma is not None:
        extra_mw = args.voltage * (args.current_ma - args.idle_current_ma)
        extra_uj = extra_mw * 1000 / rate
        report.update(idle_current_ma=args.idle_current_ma, extra_power_mw=extra_mw,
                      extra_energy_per_inference_uj=extra_uj)
        print(f"Above idle {extra_mw:.2f} mW; {extra_uj:.2f} µJ/inference")
    return report


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
    parser.add_argument("--power-benchmark", type=int, metavar="SECONDS",
                        help="Repeat one frame on the ESP32 for 1–60 seconds")
    parser.add_argument("--voltage", type=float, default=3.3, help="Measured rail voltage, default 3.3 V")
    parser.add_argument("--current-ma", type=float, help="Measured average rail current during the run")
    parser.add_argument("--idle-current-ma", type=float, help="Optional idle baseline current")
    args = parser.parse_args()
    if args.input is not None and (args.normalized is not None or args.dummy):
        parser.error("Use either a positional input, --normalized, or --dummy")
    paths = input_paths(parser, args)
    batch = args.normalized is not None and args.normalized.is_dir()

    if args.power_benchmark is not None:
        if not 1 <= args.power_benchmark <= 60 or len(paths) != 1 or batch:
            parser.error("--power-benchmark requires 1–60 seconds and a single frame or dummy input")
        if not math.isfinite(args.voltage) or args.voltage <= 0:
            parser.error("--voltage must be finite and positive")
    elif args.current_ma is not None or args.idle_current_ma is not None:
        parser.error("Current options require --power-benchmark")
    for current in (args.current_ma, args.idle_current_ma):
        if current is not None and (not math.isfinite(current) or current < 0):
            parser.error("Current must be finite and nonnegative")
    if args.idle_current_ma is not None and args.current_ma is None:
        parser.error("--idle-current-ma requires --current-ma")

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
                if args.power_benchmark is not None:
                    report = power_benchmark(link, tokens, args)
                    if path is not None:
                        report["input"] = str(path)
                    reports.append(report)
                    continue
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
