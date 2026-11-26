"""Quantize a matching floating-point SNN checkpoint and preprocessed RF frame."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from snn_reference import FORMAT, infer
from tensor_ops import LAYERS


def quantize(values, fractional, label):
    scaled = np.rint(np.asarray(values, dtype=np.float64) * (1 << fractional))
    if not np.isfinite(scaled).all() or np.any((scaled < -32768) | (scaled > 32767)):
        raise ValueError(f'{label} exceeds signed Q{fractional} range')
    return scaled.astype('<i2')


def export(weights, frame, output, event):
    params, shifts, bias_shifts, fractions = {}, [], [], []
    with np.load(weights, allow_pickle=False) as archive:
        for i, (name, shape) in enumerate(LAYERS):
            w_key, b_key = name+'.weight', name+'.bias'
            w, b = archive[w_key], archive[b_key]
            if w.shape != shape or b.shape != (shape[0],):
                raise ValueError(f'{name} requires weight {shape} and bias {(shape[0],)}')
            if w.dtype.kind != 'f' or b.dtype.kind != 'f':
                raise ValueError(f'{name} requires floating-point trained parameters')
            for fractional in range(15, (10 if i in (2, 3) else 0)-1, -1):
                try:
                    qweight = quantize(w, fractional, w_key)
                    qbias = quantize(b, fractional if i == 4 else 10, b_key)
                    break
                except ValueError:
                    continue
            else:
                raise ValueError(f'{name} cannot fit the supported fixed-point format')
            params[w_key], params[b_key] = qweight, qbias
            fractions.append(fractional)
            shift = 0 if i == 4 else fractional-10 if i in (2, 3) else fractional
            shifts.append(shift)
            bias_shifts.append(shift)

    raw = np.load(frame, allow_pickle=False)
    if raw.ndim == 4 and raw.shape[0] == 1:
        raw = raw[0]
    if raw.dtype.kind != 'f' or raw.ndim != 3 or raw.shape[1:] != (32, 156):
        raise ValueError('Expected preprocessed float [channels,32,156] or [1,channels,32,156]')
    if not 0 <= event < raw.shape[0]:
        raise ValueError('Selected channel is outside the input')
    x = quantize(raw[event:event+1], 10, 'RF input')
    head = params['head.weight'].astype(np.int64)
    bias = int(params['head.bias'][0])
    minimum = bias + int(head[head < 0].sum())
    maximum = bias + int(head[head > 0].sum())
    while minimum < -32768 or maximum > 32767:
        minimum >>= 1
        maximum >>= 1
        shifts[4] += 1
    score_fractional = fractions[-1] - shifts[4]
    config = dict(format=FORMAT, right_shifts=shifts, bias_left_shifts=bias_shifts,
                  weight_fractional_bits=fractions, score_fractional_bits=score_fractional)
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output/'weights.npz', **params)
    np.save(output/'input.npy', x)
    (output/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    score, _ = infer(params, config, x)
    print(f'Fixed-point score {score}, scaled score {math.ldexp(score, -score_fractional):.7g}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--event', type=int, default=0)
    args = parser.parse_args()
    export(args.weights, args.input, args.output, args.event)
