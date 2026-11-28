"""Benchmark ESP32 V2.1 RF pooling and inference on a reduced acquisition."""
import argparse
from binascii import crc_hqx
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import struct
from tempfile import NamedTemporaryFile
import time

import numpy as np

DEFAULT_INPUT = Path(
    r"C:\Users\ethan\Documents\Duke\Classes\Neuromorphic"
    r"\Stanford_Murine_Processed_RFData_SNNV2\skip8_V2.1_1127\unprocessed\rat01_frame00.npy"
)
DEFAULT_REPORT = Path("mcu_preprocess_power_v2.1.json")
EVENT_IDS = np.linspace(0, 383, 8, dtype=int)
CHUNK_BYTES = 512


def load_frame(path):
    frame = np.load(path, mmap_mode="r", allow_pickle=False)
    if frame.dtype.kind != "i" or frame.dtype.itemsize != 2:
        raise ValueError("Expected signed int16 RF samples in original amplitude units")
    if frame.shape == (384, 128, 1000):
        frame = frame[:, ::2, ::4]
    if frame.shape == (384, 64, 250):
        frame = frame[EVENT_IDS]
    if frame.shape == (8, 64, 256):
        if np.count_nonzero(frame[..., 250:]):
            raise ValueError("Expected the last six samples of each channel to be zero padding")
        frame = frame[..., :250]
    if frame.shape != (8, 64, 250):
        raise ValueError(f"Got RF shape {frame.shape}; expected [384,128,1000], [384,64,250], "
                         "or [8,64,250] / [8,64,256] in model event order")
    return np.ascontiguousarray(frame, dtype="<i2")


def reference_token_crc(frame):
    # Verification only. The RF bytes are the payload sent to the ESP32.
    padded = np.pad(frame.astype(np.float32), ((0, 0), (0, 0), (0, 6)))
    tokens = np.abs(padded.reshape(8, 8, 8, 16, 16)).mean(axis=(2, 4))
    return crc_hqx(tokens.transpose(2, 0, 1).astype("<f4").tobytes(order="C"), 0xffff)


def send(link, data):
    if link.write(data) != len(data):
        raise OSError("Only part of the RF payload was written")
    link.flush()


def reply(link, tag, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        line = link.readline().split()
        if line and line[0] == b"SNN_POWER_ERROR":
            raise RuntimeError(b" ".join(line[1:]).decode("ascii", errors="replace"))
        if line and line[0] == tag:
            return line[1:]
    raise TimeoutError(f"No {tag.decode()} received. Check the benchmark firmware "
                       "and reset the ESP32 before retrying.")


def benchmark(link, frame, args):
    data = frame.tobytes()
    input_crc = crc_hqx(data, 0xffff)
    expected_token_crc = reference_token_crc(frame)
    duration_ms = args.power_benchmark * 1000
    timestamp = datetime.now(timezone.utc).isoformat()
    send(link, struct.pack("<4sI", b"RF21", duration_ms))
    reply(link, b"SNN_RF_READY")
    print(f"Uploading {len(data):,} RF bytes.", flush=True)
    for offset in range(0, len(data), CHUNK_BYTES):
        end = offset + CHUNK_BYTES
        send(link, data[offset:end])
        if end < len(data):
            try:
                received, = map(int, reply(link, b"SNN_RF_RX"))
            except TimeoutError as exc:
                raise RuntimeError(f"RF upload stalled at bytes {offset + 1}–{end}. Reset the board.") from exc
            if received != end:
                raise RuntimeError(f"RF upload offset differs: got {received}, expected {end}")
    received_crc, = map(int, reply(link, b"SNN_RF_CRC"))
    if received_crc != input_crc:
        raise RuntimeError("RF payload CRC mismatch. The run should be discarded; let it finish or reset the board.")
    settle, requested, mhz = map(int, reply(link, b"SNN_POWER_START"))
    if requested != duration_ms or not 0 <= settle <= 10000 or mhz <= 0:
        raise RuntimeError("Unexpected benchmark start response")
    print(f"ESP32 CPU frequency {mhz} MHz. Inference and pooling start in {settle/1000:g} s "
          f"and run for {requested/1000:g} s.", flush=True)
    fields = reply(link, b"SNN_POWER_RESULT", (settle + requested)/1000 + 10)
    count, elapsed_us = map(int, fields[:2])
    score, actual_token_crc = float(fields[2]), int(fields[3])
    if link.read(1) != b"\xaa":
        raise RuntimeError("Missing ESP32 acknowledgment")
    if count <= 0 or elapsed_us < duration_ms * 1000 or not math.isfinite(score):
        raise RuntimeError("Invalid benchmark result")
    mean_us = elapsed_us / count
    rate = 1e6 / mean_us
    token_match = actual_token_crc == expected_token_crc
    print(f"{count} inferences in {elapsed_us/1e6:.3f} s; {rate:.2f} inferences/s; "
          f"{mean_us:.2f} µs per frame")
    print(f"Score {score:.7f}; RF CRC PASS; pooled token CRC {'PASS' if token_match else 'MISMATCH'}")
    report = dict(mode="rf_preprocess_and_snn", cpu_mhz=mhz, request_timestamp_utc=timestamp,
                  inferences=count, elapsed_us=elapsed_us, requested_seconds=args.power_benchmark,
                  mean_us_per_inference=mean_us, inferences_per_second=rate, score=score,
                  rf_bytes=len(data), input_crc_match=True, token_crc=actual_token_crc,
                  expected_token_crc=expected_token_crc, token_crc_match=token_match,
                  rail_voltage_v=args.voltage)
    if args.current_ma is not None:
        power = args.voltage * args.current_ma
        report.update(current_ma=args.current_ma, rail_power_mw=power,
                      energy_per_inference_uj=1000 * power / rate)
        print(f"Rail power {power:.3f} mW; energy {report['energy_per_inference_uj']:.3f} µJ/frame")
        if args.idle_current_ma is not None:
            extra = args.voltage * (args.current_ma - args.idle_current_ma)
            report.update(idle_current_ma=args.idle_current_ma, extra_power_mw=extra,
                          extra_energy_per_inference_uj=1000 * extra / rate)
    return report


def save_run(output, report):
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = json.loads(output.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        data = {"runs": []}
    if not isinstance(data, dict) or not isinstance(data.get("runs"), list):
        raise ValueError(f"Expected a runs list in {output}")
    data["runs"].append(report)
    temp = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                suffix=".tmp", delete=False) as file:
            temp = Path(file.name)
            json.dump(data, file, indent=2)
            file.write("\n")
        os.replace(temp, output)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)
    print(f"Saved run {len(data['runs'])} to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("port", help="ESP32 serial port, e.g. COM11")
    parser.add_argument("input", nargs="?", default=DEFAULT_INPUT, type=Path, help="RF .npy or directory")
    parser.add_argument("--power-benchmark", default=30, type=int, metavar="SECONDS")
    parser.add_argument("--output", default=DEFAULT_REPORT, type=Path, help="Append runs to this JSON file")
    parser.add_argument("--voltage", default=3.3, type=float)
    parser.add_argument("--current-ma", type=float, help="Mean active rail current from your meter")
    parser.add_argument("--idle-current-ma", type=float)
    args = parser.parse_args()
    if not 1 <= args.power_benchmark <= 60:
        parser.error("--power-benchmark must be 1–60 seconds")
    if not math.isfinite(args.voltage) or args.voltage <= 0:
        parser.error("--voltage must be finite and positive")
    for current in (args.current_ma, args.idle_current_ma):
        if current is not None and (not math.isfinite(current) or current < 0):
            parser.error("Current must be finite and nonnegative")
    if args.idle_current_ma is not None and args.current_ma is None:
        parser.error("--idle-current-ma requires --current-ma")
    paths = sorted(p for p in args.input.glob("*.npy") if p.is_file()) if args.input.is_dir() else [args.input]
    if not paths or not paths[0].is_file():
        parser.error(f"No RF input files found: {args.input}")
    if len(paths) > 1 and args.current_ma is not None:
        parser.error("Use one frame when attaching a current reading")

    import serial
    with serial.Serial(args.port, 500000, timeout=0.5, write_timeout=5) as link:
        time.sleep(2)
        link.reset_input_buffer()
        for path in paths:
            print(f"Frame {path.name}", flush=True)
            try:
                result = benchmark(link, load_frame(path), args)
                result["input"] = str(path)
                save_run(args.output, result)
            except (OSError, ValueError, RuntimeError) as exc:
                raise SystemExit(f"{path}: {exc}") from exc
            if not result["token_crc_match"]:
                raise SystemExit("ESP32 pooled tokens differ from the Python reference")


if __name__ == "__main__":
    main()
