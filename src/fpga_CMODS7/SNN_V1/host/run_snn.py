"""Check the SNN result and optional intermediate spike maps on the Cmod S7."""
import argparse
import json
import math
from pathlib import Path
import struct
from time import monotonic

import numpy as np

from snn_reference import STEPS, TRACE, infer, load_snn_model, score_from_counts
from tensor_ops import load_input, pack_weights

IDENTITY = b'\xb6\x01'


def read_exact(port, length, timeout, label):
    deadline = monotonic() + timeout
    data = bytearray()
    while len(data) < length:
        data.extend(port.read(min(4096, length - len(data))))
        if len(data) < length and monotonic() >= deadline:
            raise TimeoutError(f'{label}: got {len(data)} of {length} bytes')
    return bytes(data)


def write(port, data):
    if port.write(data) != len(data):
        raise OSError('Incomplete UART write')


def check_tag(port, expected, timeout, label):
    received = read_exact(port, len(expected), timeout, label)
    if received != expected:
        raise RuntimeError(f'{label}: expected {expected.hex()}, received {received.hex()}')


def compare(expected, actual):
    differences = np.argwhere(expected != actual)
    result = dict(match=len(differences) == 0, total_values=expected.size,
                  mismatched_values=len(differences))
    if len(differences):
        coord = tuple(int(d) for d in differences[0])
        result['first_mismatch'] = dict(coordinate=coord, expected=int(expected[coord]),
                                       actual=int(actual[coord]))
    return result


def collect(port, expected, x, continuous, timeout, params, cfg):
    expected_score, expected_trace = expected
    layers = []
    observed_counts = None
    observed_spikes3 = None
    if not continuous:
        for i, ((name, shape), ref) in enumerate(zip(TRACE, expected_trace)):
            print(f'Receiving {name} {shape}...', flush=True)
            check_tag(port, bytes([0xc0+i]), timeout, name+' tag')
            frame = read_exact(port, 2*math.prod(shape), timeout, name)
            actual = np.frombuffer(frame, dtype='<i2').reshape(shape)
            result = dict(name=name, shape=shape, **compare(ref, actual))
            if name.startswith('spikes'):
                result['spikes_per_step'] = [int(((actual >> t) & 1).sum()) for t in range(STEPS)]
            if name == 'spikes3':
                observed_spikes3 = actual
            if name == 'counts':
                observed_counts = actual
            print(f'{name} {"PASS" if result["match"] else result["first_mismatch"]}', flush=True)
            layers.append(result)
    check_tag(port, b'\xd1', timeout, 'score tag')
    score, = struct.unpack('<h', read_exact(port, 2, timeout, 'score'))
    replay = read_exact(port, x.nbytes, timeout, 'input replay')
    check_tag(port, b'\xd2', timeout, 'end tag')
    report = dict(protocol='CMODS7SNN1', layers=layers, expected_score=expected_score,
                  score=score, score_match=score == expected_score, input_replay_match=replay == x.tobytes())
    if observed_counts is not None:
        reconstructed = sum(((observed_spikes3 >> t) & 1).sum(axis=(1, 2)) for t in range(STEPS))
        report['counts_match_spikes3'] = bool(np.array_equal(reconstructed, observed_counts))
        report['score_from_counts'] = score_from_counts(params, cfg, observed_counts)
        report['head_match'] = report['score_from_counts'] == score
    report['match'] = (report['score_match'] and report['input_replay_match']
                       and all(layer['match'] for layer in layers)
                       and report.get('counts_match_spikes3', True) and report.get('head_match', True))
    return report


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', required=True)
    parser.add_argument('--weights', type=Path, default=root/'demo/demo_weights.npz')
    parser.add_argument('--config', type=Path, default=root/'demo/demo_config.json')
    parser.add_argument('--input', type=Path, default=root/'demo/demo_input.npy')
    parser.add_argument('--event', type=int, default=0, help='Input RF channel index')
    parser.add_argument('--continuous', action='store_true', help='Receive only score and input replay')
    parser.add_argument('--timeout', type=float, default=60, help='Seconds allowed per response block')
    parser.add_argument('--output', type=Path, help='Write a JSON comparison report')
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('--timeout requires a positive finite number')
    params, cfg = load_snn_model(args.weights, args.config)
    x = load_input(args.input, args.event)
    expected = infer(params, cfg, x)
    config = bytes(v for pair in zip(cfg['right_shifts'], cfg['bias_left_shifts']) for v in pair)

    import serial
    with serial.Serial(args.port, 115200, timeout=0.25, write_timeout=args.timeout) as port:
        port.reset_input_buffer()
        write(port, b'H')
        check_tag(port, IDENTITY, args.timeout, 'SNN bitstream identity')
        print('Detected SNN; uploading parameters and input.', flush=True)
        write(port, b'W'+config+pack_weights(params).tobytes())
        check_tag(port, b'\xa1', args.timeout, 'parameter acknowledgement')
        write(port, (b'I' if args.continuous else b'T') + x.tobytes())
        report = collect(port, expected, x, args.continuous, args.timeout, params, cfg)

    print(f'FPGA score {report["score"]}; reference {report["expected_score"]}; '
          f'input replay {"PASS" if report["input_replay_match"] else "MISMATCH"}')
    if 'score_fractional_bits' in cfg:
        print(f'Scaled logit {math.ldexp(report["score"], -cfg["score_fractional_bits"]):.7g}')
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    if not report['match']:
        raise SystemExit('SNN comparison failed')


if __name__ == '__main__':
    main()
