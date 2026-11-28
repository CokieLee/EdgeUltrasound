"""Check UPduino SNN V2 inference through the onboard FTDI."""
# to run:
# $tokens = 'C:\Users\ethan\Documents\Duke\Classes\Neuromorphic\Stanford_Murine_Processed_RFData_SNNV2\Tokens\float32'
# One real frame with per-token checks
# python host/run_snn_v2.py --port COM7 --normalized "$tokens\rat01_frame00.npy" --trace
# Whole folder in filename order, with one report for all frames
# python host/run_snn_v2.py --port COM7 --normalized "$tokens" --output fpga_real.json
# Original synthetic smoke test, expected Q20 score -1055882
# python host/run_snn_v2.py --port COM7 --dummy --trace

import argparse
from binascii import crc_hqx
from contextlib import nullcontext
import json
from pathlib import Path
import struct
from time import monotonic

import numpy as np

from reference import (SCORE_SCALE, demo_tokens, encode_rf, infer_fixed, infer_float,
                       load_weights, normalize, quantize)

ROOT = Path(__file__).resolve().parents[1]


def read_exact(port, size, label):
    deadline = monotonic()+5
    result = bytearray()
    while len(result)<size:
        result.extend(port.read(size-len(result)))
        if monotonic()>=deadline and len(result)<size:
            raise TimeoutError(f'{label}: received {len(result)} of {size} bytes')
    return bytes(result)


def send(port, payload):
    if port.write(payload) != len(payload):
        raise OSError('Incomplete write')


def require_tag(port, tag, label):
    actual = read_exact(port, 1, label)[0]
    if actual != tag:
        detail = 'FPGA rejected the request' if actual == 0xee else f'got {actual:02x}, expected {tag:02x}'
        raise ValueError(f'{label}: {detail}')


def trace_result(port, t, expected):
    require_tag(port, 0xc0+t, f'token {t} trace')
    data = read_exact(port, 66, f'token {t} trace')
    spike1 = np.unpackbits(np.frombuffer(data[:1], dtype=np.uint8), bitorder='little').astype(bool)
    spike2 = np.unpackbits(np.frombuffer(data[1:2], dtype=np.uint8), bitorder='little').astype(bool)
    v = np.frombuffer(data[2:], dtype='<i4').reshape(2, 8)
    match = True
    for name, actual, ref in zip(('spikes1','spikes2','v1','v2'), (spike1, spike2, *v), expected):
        bad = np.flatnonzero(actual!=ref)
        if bad.size:
            i = int(bad[0])
            print(f'Token {t} {name}[{i}]: FPGA {actual[i]}, expected {ref[i]}')
            match = False
    return dict(token=t, match=match, spikes1=int(data[0]), spikes2=int(data[1]),
                voltage1_q18=v[0].tolist(), voltage2_q18=v[1].tolist())


def exchange(port, q, config, expected_trace, trace):
    send(port, b'H')
    identity = read_exact(port, 10, 'model identity')
    expected = b'\xb7\x01'+bytes.fromhex(config['weights_sha256'][:16])
    if identity != expected:
        raise RuntimeError(f'Wrong FPGA bitstream or checkpoint: received {identity.hex()}, expected {expected.hex()}')
    send(port, b'T' if trace else b'I')
    steps = []
    for t, token in enumerate(q):
        send(port, token.tobytes())
        if trace:
            steps.append(trace_result(port, t, expected_trace[t]))
        if t != 15:
            require_tag(port, 0xa1, f'token {t} credit')
    require_tag(port, 0xd1, 'result')
    score, checksum = struct.unpack('<iH', read_exact(port, 6, 'score and CRC'))
    return score, checksum, steps



def input_paths(args):
    """Resolve one input or a flat folder of normalized frames."""
    source = args.normalized or args.tokens or args.rf
    if source is None:
        return [None]
    if args.normalized and source.is_dir():
        paths = sorted(p for p in source.glob('*.npy') if p.is_file())
        if not paths:
            raise ValueError(f'{source} contains no .npy files')
        return paths
    if not source.is_file():
        raise ValueError(f'Input file not found: {source}')
    return [source]


def load_frame(source, args, config):
    if source is None:
        print('Using synthetic tokens with the supplied trained checkpoint.')
        return demo_tokens()
    try:
        array = np.load(source, allow_pickle=False)
        if args.rf:
            return normalize(encode_rf(array), config)
        return normalize(array, config, normalized=args.normalized is not None)
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f'{source}: {exc}') from exc


def run_frame(x, cfg, p, fixed, port, trace):

    q = quantize(x)
    ref, expected_trace = infer_fixed(fixed, q)
    floating_score, _ = infer_float(p, x)
    report = dict(python_score_q20=ref, python_score=ref/SCORE_SCALE, float_score=floating_score,
                  fixed_minus_float=ref/SCORE_SCALE-floating_score)
    print(f'Python fixed-point score {ref} ({ref/SCORE_SCALE:.8f}); float score {floating_score:.8f}')
    if port is not None:
        score, checksum, steps = exchange(port, q, cfg, expected_trace, trace)
        crc_match = checksum == crc_hqx(q.tobytes(), 0xffff)
        report.update(fpga_score_q20=score, fpga_score=score/SCORE_SCALE, score_match=score==ref,
                      input_crc_match=crc_match, trace=steps,
                      match=score==ref and crc_match and all(step['match'] for step in steps))
        print(f'FPGA score {score} ({score/SCORE_SCALE:.8f}); input CRC {"PASS" if crc_match else "MISMATCH"}')
        if trace:
            print(f'Per-token spikes and voltages: {"PASS" if all(step["match"] for step in steps) else "MISMATCH"}')
    standardized = report.get('fpga_score', report['python_score'])
    report['predicted_target_percent'] = cfg['target_mean']+cfg['target_scale']*standardized
    print(f'Model output in target units {report["predicted_target_percent"]:.8f}%')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--ftdi", nargs="?", const="ftdi://ftdi:232h/1", help="Use the onboard FT232H, optionally with a PyFtdi URL")
    mode.add_argument('--reference-only', action='store_true')
    data = parser.add_mutually_exclusive_group()
    data.add_argument('--rf', type=Path, help='Original unnormalized [384,64,250] RF frame .npy')
    data.add_argument('--tokens', type=Path, help='Encoded [16,64] mean-amplitude tokens .npy before gain')
    data.add_argument('--normalized', type=Path,
                      help='Normalized [16,64] .npy in [0,5], or a folder of those files')
    data.add_argument('--dummy', action='store_true', help='Send the built-in synthetic test frame (default)')
    parser.add_argument('--trace', action='store_true', help='Check per-token spike masks and voltages')
    parser.add_argument('--output', type=Path, help='Write results to JSON')
    args = parser.parse_args()
    try:
        paths = input_paths(args)
    except ValueError as exc:
        parser.error(str(exc))
    cfg, p, fixed = load_weights(ROOT/'weights')

    port_context = nullcontext()
    if args.ftdi:
        from ftdi_spi import FtdiSpi
        port_context = FtdiSpi(args.ftdi)

    reports = []
    with port_context as port:
        for source in paths:
            if source is not None:
                print(f'Frame {source.name}', flush=True)
            try:
                x = load_frame(source, args, cfg)
                report = run_frame(x, cfg, p, fixed, port, args.trace)
            except (OSError, ValueError, RuntimeError) as exc:
                raise SystemExit(f'{source.name if source else "Dummy frame"}: {exc}') from exc
            if source is not None:
                report['input'] = str(source)
            reports.append(report)

    passed = all(report.get('match', True) for report in reports)
    batch = args.normalized is not None and args.normalized.is_dir()
    if batch:
        print(f'Processed {len(reports)} frames.')
    if args.output:
        output = dict(frames=reports, match=passed) if batch else reports[0]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2)+'\n')
    if not passed:
        raise SystemExit('FPGA comparison failed')


if __name__ == '__main__':
    main()
