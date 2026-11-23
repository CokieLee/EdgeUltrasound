#!/usr/bin/env python3
"""Read produced layer words and raw pooling sums on CNNSTREAM2 or later."""
import argparse
from datetime import datetime
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np

from cnn_reference import conv, infer, load_input, load_model, pack_weights, requantize
from run_cnn import read_exact, send_all

STAGES = [('stem', (8, 32, 156)), ('conv1', (16, 16, 78)),
          ('conv2', (32, 8, 39)), ('conv3', (32, 4, 20)), ('pool', (32,))]


def comparison(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape:
        raise ValueError('Tensor shapes differ')
    diff = actual.astype(np.int64) - expected.astype(np.int64)
    indices = np.flatnonzero(diff)
    result = {'match': indices.size == 0, 'total_values': int(actual.size),
              'mismatched_values': int(indices.size), 'max_abs_error': int(np.abs(diff).max()),
              'mean_signed_error': float(diff.mean())}
    if indices.size:
        pos = np.unravel_index(int(indices[0]), actual.shape)
        result['first_mismatch'] = {'coordinate': [int(v) for v in pos],
                                   'expected': int(expected[pos]), 'actual': int(actual[pos])}
    return result


def divide_pool_sums(sums):
    # SystemVerilog signed division truncates toward zero.
    return np.where(sums >= 0, sums // 80, -((-sums) // 80)).astype('<i2')


def actual_pool(x):
    return divide_pool_sums(x.astype(np.int64).sum(axis=(1, 2)))


class CapturePort:
    def __init__(self, stream):
        self.stream = stream

    def read(self, count):
        result = self.stream.read(count)
        if not result:
            raise EOFError('Capture ended before the response was complete')
        return result

    def write(self, data):
        return len(data)


class RecordPort:
    def __init__(self, port, output):
        self.port, self.output = port, output

    def read(self, count):
        result = self.port.read(count)
        self.output.write(result)
        self.output.flush()
        return result

    def write(self, data):
        return self.port.write(data)


def collect(port, params, cfg, x, output, timeout=60, streaming=True, require_pool_sums=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    expected_score, expected_layers = infer(params, cfg, x, True)
    report = {'protocol': None, 'completed': False, 'streaming': streaming,
              'snapshot_kind': 'Produced output words; not independent RAM readback',
              'parameter_readback_verified': False, 'layer_transport_crc': False,
              'layers': [], 'expected_score': expected_score,
              'first_reference_mismatch': None, 'first_local_mismatch': None}

    def save():
        (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')

    try:
        print('Checking minimal-stream identity...', flush=True)
        send_all(port, b'H')
        identity = read_exact(port, 2, min(timeout, 5))
        if identity not in (b'\xb5\x01', b'\xb5\x02', b'\xb5\x03'):
            raise RuntimeError(f'Unexpected bitstream identity {identity.hex()}')
        has_pool_sums = identity[1] >= 2
        report['protocol'] = f'CNNSTREAM{identity[1]}'
        print(f'Detected {report["protocol"]}', flush=True)
        if require_pool_sums and not has_pool_sums:
            raise RuntimeError('Raw pooling sums require CNNSTREAM2 or CNNSTREAM3')
        config = bytes(v for pair in zip(cfg['right_shifts'], cfg['bias_left_shifts']) for v in pair)
        print('Uploading original weights/config and input...', flush=True)
        send_all(port, b'W' + config + pack_weights(params).tobytes())
        if read_exact(port, 1, timeout) != b'\xa1':
            raise RuntimeError('Weight acknowledgment invalid')
        payload = x.astype('<i2', copy=False).tobytes()
        send_all(port, (b'T' if streaming else b'I') + payload)
        previous = x
        pooled = None
        for i, (name, shape) in enumerate(STAGES if streaming else []):
            print(f'Receiving {name} {shape}...', flush=True)
            if read_exact(port, 1, timeout) != bytes([0xc0 + i]):
                raise RuntimeError(f'Unexpected layer tag before {name}')
            if name == 'pool' and has_pool_sums:
                payload_pool = read_exact(port, shape[0] * 8, timeout)
                actual = np.frombuffer(payload_pool, dtype='<i2')[::4].copy()
                sums = np.array([
                    int.from_bytes(payload_pool[offset + 2:offset + 8], 'little', signed=True)
                    for offset in range(0, len(payload_pool), 8)
                ], dtype=np.int64)
                expected_sums = previous.astype(np.int64).sum(axis=(1, 2))
                quotients = divide_pool_sums(sums)
                np.save(output / 'pool_sums.npy', sums)
                sum_check = comparison(sums, expected_sums)
                quotient_check = comparison(actual, quotients)
                report['pool_diagnostic'] = {
                    'sum_vs_emitted_conv3': sum_check,
                    'quotient_vs_fpga_sum_divided_by_80': quotient_check,
                    'channels': [
                        {'channel': i, 'expected_sum': int(expected_sums[i]),
                         'fpga_sum': int(sums[i]), 'fpga_pool': int(actual[i]),
                         'expected_pool_from_fpga_sum': int(quotients[i])}
                        for i in range(shape[0])
                    ],
                }
                for label, check in (('Raw sums vs. conv3', sum_check),
                                     ('Pool outputs vs. raw sums / 80', quotient_check)):
                    print(f'{label}: {"PASS" if check["match"] else "MISMATCH"}', flush=True)
                    if not check['match']:
                        print('  First difference:', check['first_mismatch'], flush=True)
            else:
                actual = np.frombuffer(read_exact(port, int(np.prod(shape)) * 2, timeout),
                                       '<i2').reshape(shape).copy()
            np.save(output / (name + '.npy'), actual)
            reference = comparison(actual, expected_layers[name])
            if name == 'pool':
                local_expected = actual_pool(previous)
                pooled = actual
            else:
                local_expected = conv(previous, params[name + '.weight'], params[name + '.bias'],
                    1 if i == 0 else 2, 0 if i == 0 else 1,
                    cfg['right_shifts'][i], cfg['bias_left_shifts'][i], i != 0)
            local = comparison(actual, local_expected)
            report['layers'].append({'name': name, 'shape': list(shape),
                'reference_comparison': reference,
                'given_emitted_input_and_sent_parameters': local})
            if not reference['match'] and report['first_reference_mismatch'] is None:
                report['first_reference_mismatch'] = name
            if not local['match'] and report['first_local_mismatch'] is None:
                report['first_local_mismatch'] = name
            print(f'{name}: reference {"PASS" if reference["match"] else "MISMATCH"}; '
                  f'using emitted input/sent parameters {"PASS" if local["match"] else "MISMATCH"}', flush=True)
            if not reference['match']:
                print('  First difference:', reference['first_mismatch'], flush=True)
            previous = actual
            save()
        if read_exact(port, 1, timeout) != b'\xd1':
            raise RuntimeError('Final inference marker invalid')
        score = struct.unpack('<h', read_exact(port, 2, timeout))[0]
        replay = read_exact(port, len(payload), timeout)
        if read_exact(port, 1, timeout) != b'\xd2':
            raise RuntimeError('Final input-replay marker invalid')
        np.save(output / 'input_replay.npy', np.frombuffer(replay, '<i2').reshape(x.shape))
        report['input_replay_match'] = replay == payload
        report['score'] = {'actual': score, 'expected': expected_score, 'match': score == expected_score}
        if pooled is not None:
            acc = params['head.weight'].astype(np.int64) @ pooled.astype(np.int64)
            acc += params['head.bias'].astype(np.int64) << cfg['bias_left_shifts'][4]
            predicted = int(requantize(acc, cfg['right_shifts'][4], False)[0])
            report['score']['from_emitted_pool_and_sent_head'] = predicted
            report['score']['local_match'] = score == predicted
            if score != predicted and report['first_local_mismatch'] is None:
                report['first_local_mismatch'] = 'head'
        if score != expected_score and report['first_reference_mismatch'] is None:
            report['first_reference_mismatch'] = 'head'
        report['completed'] = True
        print(f'Score: FPGA {score}, reference {expected_score}; input replay '
              f'{"PASS" if report["input_replay_match"] else "MISMATCH"}', flush=True)
    except (Exception, KeyboardInterrupt) as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        print(f'ABORTED: {exc}. Reset button 0 before retrying.', file=sys.stderr)
    finally:
        save()
    return report


def passed(report):
    diagnostic = report.get('pool_diagnostic')
    return (report['completed'] and report['input_replay_match'] and report['score']['match']
            and report['first_reference_mismatch'] is None
            and report['first_local_mismatch'] is None
            and (diagnostic is None or all(diagnostic[key]['match'] for key in (
                'sum_vs_emitted_conv3', 'quotient_vs_fpga_sum_divided_by_80'))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--port')
    source.add_argument('--capture', type=Path)
    parser.add_argument('--weights', type=Path, default=Path('demo/demo_weights.npz'))
    parser.add_argument('--config', type=Path, default=Path('demo/demo_config.json'))
    parser.add_argument('--input', type=Path, default=Path('demo/demo_input.npy'))
    parser.add_argument('--event', type=int, default=0)
    parser.add_argument('--timeout', type=float, default=60)
    parser.add_argument('--continuous', action='store_true')
    parser.add_argument('--require-pool-sums', action='store_true',
                        help='Fail early if the FPGA or capture has the older stream protocol')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    if args.continuous and args.require_pool_sums:
        parser.error('--require-pool-sums needs streaming mode')
    output = args.output or Path('stream_results') / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error('--output must be fresh or empty')
    output.mkdir(parents=True, exist_ok=True)
    params, cfg = load_model(args.weights, args.config)
    x = load_input(args.input, args.event)
    manifest = {'source': str(args.capture) if args.capture else args.port, 'event': args.event,
                'sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                           (args.weights, args.config, args.input)}}
    (output / 'host_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    if args.capture:
        with args.capture.open('rb') as capture:
            report = collect(CapturePort(capture), params, cfg, x, output,
                             args.timeout, not args.continuous, args.require_pool_sums)
    else:
        import serial
        with serial.Serial(args.port, 115200, timeout=0.05, write_timeout=args.timeout,
                           xonxoff=False, rtscts=False, dsrdtr=False) as port:
            port.reset_input_buffer()
            with (output / 'rx.bin').open('wb') as capture:
                report = collect(RecordPort(port, capture), params, cfg, x, output,
                                 args.timeout, not args.continuous, args.require_pool_sums)
    print('Results:', output.resolve())
    print('First reference mismatch:', report['first_reference_mismatch'])
    print('First local mismatch:', report['first_local_mismatch'])
    return 0 if passed(report) else 1


if __name__ == '__main__':
    raise SystemExit(main())
