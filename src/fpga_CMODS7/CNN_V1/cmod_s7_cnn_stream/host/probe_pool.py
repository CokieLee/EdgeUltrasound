#!/usr/bin/env python3
"""Compare equal-sum conv3 layouts on the existing CNNSTREAM1 bitstream."""
import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np

from cnn_reference import LAYERS, infer, load_input, load_model
from run_stream import RecordPort, collect


def make_probes(conv3_channel):
    # Three stride-2 center taps route input[::8, ::8] into conv3.
    params = {}
    for name, shape in LAYERS:
        params[name + '.weight'] = np.zeros(shape, dtype='<i2')
        params[name + '.bias'] = np.zeros(shape[0], dtype='<i2')
    params['stem.weight'][0, 0, 0, 0] = 1
    for name in ('conv1', 'conv2'):
        params[name + '.weight'][0, 0, 1, 1] = 1
    params['conv3.weight'][:, 0, 1, 1] = 1
    params['head.weight'][0, 0] = 1
    # Channel 0 uses the unmodified map. Offsets make wrong-channel reads visible.
    offsets = np.arange(32, dtype=np.int64) * 13
    values = conv3_channel.reshape(-1).astype(np.int64)
    # Keep the exact spatial profile for the demo while avoiding saturation
    # if the caller supplies another model with higher activations.
    offsets = np.minimum(offsets, 32767 - int(values.max()))
    params['conv3.bias'][:] = offsets
    cfg = {'right_shifts': [0] * 5, 'bias_left_shifts': [0] * 5}

    average, remainder = divmod(int(values.sum()), values.size)
    nearly_flat = np.full(values.shape, average, dtype=np.int64)
    nearly_flat[:remainder] += 1
    order = np.random.default_rng(23).permutation(values.size)
    for name, layout in (('original_layout', values),
                         ('shuffled_layout', values[order]),
                         ('nearly_flat_layout', nearly_flat)):
        grid = layout.reshape(4, 20)
        x = np.zeros((1, 32, 156), dtype='<i2')
        x[0, ::8, ::8] = grid
        expected_maps = grid[None, :, :] + offsets[:, None, None]
        yield name, params, cfg, x, expected_maps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', required=True)
    parser.add_argument('--weights', type=Path, default=Path('demo/demo_weights.npz'))
    parser.add_argument('--config', type=Path, default=Path('demo/demo_config.json'))
    parser.add_argument('--input', type=Path, default=Path('demo/demo_input.npy'))
    parser.add_argument('--event', type=int, default=0)
    args = parser.parse_args()
    params, cfg = load_model(args.weights, args.config)
    _, layers = infer(params, cfg, load_input(args.input, args.event), return_layers=True)
    probes = list(make_probes(layers['conv3'][0]))
    output = Path('stream_results') / datetime.now().strftime('pool_probe_%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True)
    summary = {'source_demo': {'weights': str(args.weights), 'config': str(args.config),
                               'input': str(args.input), 'event': args.event},
               'cases': {}, 'completed': False}
    summary_path = output / 'probe_summary.json'

    import serial
    try:
        with serial.Serial(args.port, 115200, timeout=0.05, write_timeout=60,
                           xonxoff=False, rtscts=False, dsrdtr=False) as uart:
            uart.reset_input_buffer()
            for name, weights, shifts, x, expected_maps in probes:
                folder = output / name
                folder.mkdir()
                print(f'\nRunning {name}...', flush=True)
                with (folder / 'rx.bin').open('wb') as capture:
                    report = collect(RecordPort(uart, capture), weights, shifts, x, folder)
                entry = summary['cases'][name] = {'completed': report['completed']}
                if not report['completed']:
                    entry['error'] = report.get('error')
                    break
                observed_maps = np.load(folder / 'conv3.npy', allow_pickle=False)
                pool = np.load(folder / 'pool.npy', allow_pickle=False)
                sums = observed_maps.astype(np.int64).sum(axis=(1, 2))
                entry.update({'conv3_matches_probe': bool(np.array_equal(observed_maps, expected_maps)),
                              'input_replay_matches': report['input_replay_match'],
                              'conv3_sums': sums.tolist(), 'fpga_pool': pool.tolist(),
                              'expected_pool': (expected_maps.sum(axis=(1, 2)) // 80).tolist()})
                print(f'Channel 0: conv3 sum {sums[0]}, FPGA pool {pool[0]}, '
                      f'expected {entry["expected_pool"][0]}', flush=True)
            summary['completed'] = (len(summary['cases']) == len(probes)
                                    and all(entry['completed'] for entry in summary['cases'].values()))
        cases = list(summary['cases'].values())
        if summary['completed'] and all(c['conv3_matches_probe'] and c['input_replay_matches']
                                        for c in cases):
            summary['different_pool_outputs_despite_equal_sums'] = [
                i for i, (a, b, c) in enumerate(zip(*(case['fpga_pool'] for case in cases)))
                if a != b or a != c]
            print('Channels with layout-dependent pooled values:',
                  summary['different_pool_outputs_despite_equal_sums'])
    finally:
        summary_path.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
        print(f'Summary: {summary_path}')
    return 0 if summary['completed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
