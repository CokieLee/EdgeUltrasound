"""Time-major fixed-point reference for the Cmod S7 SNN."""
import argparse
import json
from pathlib import Path

import numpy as np

from tensor_ops import LAYERS, conv, load_model, requantize

FORMAT = 'SNN4_Q10_V1'
STEPS = 4
THRESHOLD = 1024
BETA = 29491
TRACE = (
    ('stem', (8, 32, 156)),
    ('spikes1', (16, 16, 78)),
    ('spikes2', (32, 8, 39)),
    ('spikes3', (32, 4, 20)),
    ('counts', (32,)),
)


def load_snn_model(weights, config):
    params, cfg = load_model(weights, config)
    if cfg.get('format') != FORMAT or cfg['bias_left_shifts'][4] != 0:
        raise ValueError(f'Expected {FORMAT} with zero head bias left shift')
    return params, cfg


def lif(previous, current):
    reset = previous > THRESHOLD
    membrane = (previous * BETA >> 15) + current - THRESHOLD * reset
    return membrane > THRESHOLD, membrane


def infer(params, cfg, x):
    if x.shape != (1, 32, 156) or x.dtype.kind != 'i' or x.dtype.itemsize != 2:
        raise ValueError('Expected input shape (1,32,156) and signed int16')

    def convolution(i, source):
        name = LAYERS[i][0]
        return conv(source, params[name+'.weight'], params[name+'.bias'],
                    1 if i == 0 else 2, 0 if i == 0 else 1,
                    cfg['right_shifts'][i], cfg['bias_left_shifts'][i], False)

    stem = convolution(0, x)
    current1 = convolution(1, stem)
    packed = [np.zeros(shape, dtype=np.int16) for _, shape in TRACE[1:4]]
    membranes = [np.zeros(shape, dtype=np.int64) for _, shape in TRACE[1:4]]
    counts = np.zeros(32, dtype=np.int64)
    for t in range(STEPS):
        spike = None
        for i in range(3):
            current = current1 if i == 0 else convolution(i + 1, spike)
            spike, membranes[i] = lif(membranes[i], current)
            packed[i] |= spike.astype(np.int16) << t
        counts += spike.sum(axis=(1, 2))

    return score_from_counts(params, cfg, counts), [stem, *packed, counts.astype(np.int16)]


def score_from_counts(params, cfg, counts):
    weights = params['head.weight'][0].astype(np.int64)
    bias = int(params['head.bias'][0])
    numerator = int(weights @ counts.astype(np.int64)) + 320 * bias
    return int(requantize(numerator // 320, cfg['right_shifts'][4], False))


def make_demo(folder):
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    params = {}
    for i, (name, shape) in enumerate(LAYERS):
        extent = (1024, 95, 130, 155, 390)[i]
        params[name+'.weight'] = rng.integers(-extent, extent+1, shape, dtype=np.int16)
        params[name+'.bias'] = rng.integers(150, 950, shape[0], dtype=np.int16)
    params['head.bias'][0] = -450
    cfg = dict(format=FORMAT, right_shifts=[10, 10, 0, 0, 0],
               bias_left_shifts=[10, 10, 0, 0, 0],
               description='Untrained synthetic hardware smoke test')
    x = rng.integers(-2100, 2101, (1, 32, 156), dtype=np.int16)
    np.savez(folder/'demo_weights.npz', **params)
    (folder/'demo_config.json').write_text(json.dumps(cfg, indent=2)+'\n')
    np.save(folder/'demo_input.npy', x)
    return params, cfg, x


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--make-demo', type=Path, required=True)
    args = parser.parse_args()
    score, layers = infer(*make_demo(args.make_demo))
    print(f'Synthetic reference score {score}')
    for (name, _), layer in zip(TRACE[1:4], layers[1:4]):
        print(name, [int(((layer >> t) & 1).sum()) for t in range(STEPS)])
