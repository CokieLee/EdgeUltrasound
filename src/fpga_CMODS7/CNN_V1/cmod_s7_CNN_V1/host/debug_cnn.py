#!/usr/bin/env python3
"""Diagnose the existing CNN bitstream using known-output runtime models.

No new FPGA protocol or bitstream is required. Each request reloads all weights
and shifts, checks every replayed input byte, and records score mismatches.
"""
import argparse
import hashlib
import json
import platform
import struct
import sys
import time
from pathlib import Path

import numpy as np

from cnn_reference import LAYERS, infer, load_input, load_model, pack_weights
from run_cnn import read_exact, send_all


def blank_model():
    params = {}
    for name, shape in LAYERS:
        params[name + '.weight'] = np.zeros(shape, dtype='<i2')
        params[name + '.bias'] = np.zeros(shape[0], dtype='<i2')
    cfg = {'right_shifts': [0] * 5, 'bias_left_shifts': [0] * 5}
    return params, cfg


def path_model(high=False):
    """Propagate one channel through the center tap of each convolution."""
    params, cfg = blank_model()
    channels = [7, 15, 31, 31] if high else [0, 0, 0, 0]
    previous = channels[0]
    for name, current in zip(('conv1', 'conv2', 'conv3'), channels[1:]):
        params[name + '.weight'][current, previous, 1, 1] = 1
        previous = current
    params['head.weight'][0, previous] = 1
    return params, cfg, channels[0]


def smoke_cases(params, cfg, x):
    # Each tuple has a separate, simple expected answer as well as the NumPy
    # network reference. This avoids relying only on two implementations agreeing.
    yield 'baseline', params, cfg, None
    for value in (123, -123):
        p, c = blank_model()
        p['head.bias'][0] = value
        yield 'head_bias_' + str(value), p, c, value

    p, c, channel = path_model()
    p['stem.bias'][channel] = 64
    yield 'constant_low_channel', p, c, 64

    p, c, channel = path_model(high=True)
    p['stem.bias'][channel] = 1
    c['bias_left_shifts'][0] = 6
    yield 'constant_high_channels_bias_shift', p, c, 64

    p, c, channel = path_model()
    p['stem.bias'][channel] = 67
    p['head.weight'][0, 0] = -1
    c['right_shifts'] = [1] * 5
    # 67 -> 33 -> 16 -> 8 -> 4; mean=4; (-1 * 4) >> 1 = -2.
    yield 'signed_head_and_right_shifts', p, c, -2

    p, c, channel = path_model()
    p['stem.weight'][channel, 0, 0, 0] = 1
    # Three stride-2 center-tap layers select every eighth row/sample.
    expected = int(np.maximum(x[0, ::8, ::8].astype(np.int64), 0).sum() // 80)
    yield 'input_center_path', p, c, expected


def transact(port, params, cfg, x, timeout, label):
    config_bytes = bytes(v for pair in zip(cfg['right_shifts'],
                                         cfg['bias_left_shifts']) for v in pair)
    payload = np.ascontiguousarray(x, dtype='<i2').tobytes()
    phase = 'uploading weights/config'
    try:
        print(f'[{label}] {phase}', flush=True)
        send_all(port, b'W' + config_bytes + pack_weights(params).tobytes())
        phase = 'waiting for weight acknowledgment'
        if read_exact(port, 1, timeout) != b'\xa1':
            raise RuntimeError('Invalid weight acknowledgment')
        phase = 'uploading input'
        send_all(port, b'I' + payload)
        phase = 'waiting for inference response'
        if read_exact(port, 1, timeout) != b'\xd1':
            raise RuntimeError('Invalid inference response marker')
        phase = 'reading score and original input replay'
        actual = struct.unpack('<h', read_exact(port, 2, timeout))[0]
        replay = read_exact(port, len(payload), timeout)
        if read_exact(port, 1, timeout) != b'\xd2':
            raise RuntimeError('Invalid final response marker')
        if replay != payload:
            offset = next(i for i, (a, b) in enumerate(zip(replay, payload)) if a != b)
            raise RuntimeError(f'Input replay mismatch at byte {offset}')
        return actual
    except Exception as exc:
        raise RuntimeError(f'{label}: {phase}: {exc}. Stop, reset button 0, '
                           'and rerun; do not resume a partial transaction.') from exc


def probe_models(params, cfg):
    for channel in range(32):
        p = {k: v.copy() for k, v in params.items()}
        c = {k: list(cfg[k]) for k in ('right_shifts', 'bias_left_shifts')}
        p['head.weight'][:] = 0
        p['head.weight'][0, channel] = 1
        p['head.bias'][:] = 0
        c['right_shifts'][4] = 0
        c['bias_left_shifts'][4] = 0
        yield channel, p, c


def head_from_pool(params, cfg, pool):
    acc = sum(int(w) * int(v) for w, v in zip(params['head.weight'][0], pool))
    acc += int(params['head.bias'][0]) << cfg['bias_left_shifts'][4]
    return max(-32768, min(32767, acc >> cfg['right_shifts'][4]))


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', help='Windows COM port or Linux /dev/ttyUSB port')
    parser.add_argument('--weights', type=Path, default=Path('demo/demo_weights.npz'))
    parser.add_argument('--config', type=Path, default=Path('demo/demo_config.json'))
    parser.add_argument('--input', type=Path, default=Path('demo/demo_input.npy'))
    parser.add_argument('--event', type=int, default=0)
    parser.add_argument('--timeout', type=float, default=60)
    parser.add_argument('--probe-pool', action='store_true',
                        help='Also run 32 one-hot-head inferences (several minutes)')
    parser.add_argument('--reference-only', action='store_true',
                        help='Check expected cases without opening a serial port')
    parser.add_argument('--report', type=Path, default=Path('cnn_debug_report.json'))
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    if not args.reference_only and not args.port:
        parser.error('--port is required unless --reference-only is used')
    params, cfg = load_model(args.weights, args.config)
    x = load_input(args.input, args.event)
    expected_score, layers = infer(params, cfg, x, return_layers=True)
    cases = []
    for name, p, c, independent in smoke_cases(params, cfg, x):
        expected = infer(p, c, x)
        if independent is not None and expected != independent:
            raise RuntimeError(f'Reference self-check failed for {name}: '
                               f'{expected} != {independent}')
        cases.append((name, p, c, expected))
    if args.reference_only:
        for name, _, _, expected in cases:
            print(f'{name}: expected {expected}')
        return 0

    report = {
        'port': args.port, 'event': args.event, 'python': sys.version,
        'platform': platform.platform(), 'numpy': np.__version__,
        'config': cfg, 'expected_score': expected_score,
        'files_sha256': {str(p): file_hash(p) for p in
                         (args.weights, args.config, args.input, Path(__file__),
                          Path(__file__).with_name('cnn_reference.py'),
                          Path(__file__).with_name('run_cnn.py'))},
        'tests': [], 'completed': False,
    }

    def save_report():
        args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')

    print('Use the existing CNN bitstream. Reset button 0 BEFORE running this script.')
    print('Seven tests; each reloads all parameters. Do not reset during the run.', flush=True)
    try:
        import serial
        with serial.Serial(args.port, 115200, timeout=0.05, write_timeout=args.timeout,
                           xonxoff=False, rtscts=False, dsrdtr=False) as port:
            port.reset_input_buffer()
            for name, p, c, expected in cases:
                started = time.monotonic()
                actual = transact(port, p, c, x, args.timeout, name)
                result = {'name': name, 'expected': expected, 'actual': actual,
                          'match': actual == expected, 'input_replay_match': True,
                          'elapsed_seconds': round(time.monotonic() - started, 3)}
                report['tests'].append(result)
                save_report()
                print(f'{"PASS" if result["match"] else "MISMATCH"} {name}: '
                      f'expected {expected}, FPGA {actual}; input replay matches', flush=True)
            if args.probe_pool:
                print('Pool probes use the head as a measurement path, not raw RAM readback.',
                      flush=True)
                print('Each probe reloads parameters; interpret with the known-output tests.',
                      flush=True)
                pool = []
                report['pool_probes'] = []
                for channel, p, c in probe_models(params, cfg):
                    expected = int(layers['pool'][channel])
                    actual = transact(port, p, c, x, args.timeout, f'pool[{channel}]')
                    pool.append(actual)
                    report['pool_probes'].append({'channel': channel, 'expected': expected,
                                                 'actual': actual, 'match': actual == expected,
                                                 'input_replay_match': True})
                    save_report()
                    print(f'pool[{channel}]: expected {expected}, FPGA {actual}', flush=True)
                reconstructed = head_from_pool(params, cfg, pool)
                report['score_from_probed_pool'] = reconstructed
                print(f'Original head applied in Python to probed pool: {reconstructed}; '
                      f'original reference score: {expected_score}', flush=True)
            report['completed'] = True
    except (Exception, KeyboardInterrupt) as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        print(f'ABORTED: {exc}', file=sys.stderr)
    finally:
        save_report()
        print(f'Report saved: {args.report}', flush=True)
    if not report['completed']:
        return 2
    return int(any(not t['match'] for t in report['tests'] + report.get('pool_probes', [])))


if __name__ == '__main__':
    raise SystemExit(main())
