"""RF encoder, floating-point baseline, and bit-exact FPGA arithmetic."""
import json
from pathlib import Path

import numpy as np

from export_weights import KEYS

EVENTS = np.linspace(0, 383, 8, dtype=int)
INPUT_SCALE = 1 << 12
VOLTAGE_SCALE = 1 << 18
SCORE_SCALE = 1 << 20


def load_weights(folder):
    cfg = json.loads((Path(folder)/'config.json').read_text())
    with np.load(Path(folder)/'reference.npz', allow_pickle=False) as data:
        floating = {k: data[k] for k in KEYS}
        fixed = {k: data['q_'+k].astype(np.int64) for k in KEYS}
    return cfg, floating, fixed


def encode_rf(frame):
    if frame.shape != (384, 64, 250):
        raise ValueError('RF input requires one unnormalized [384,64,250] frame')
    x = frame[EVENTS].astype(np.float32)
    x = np.pad(x, ((0, 0), (0, 0), (0, 6)))
    return np.abs(x.reshape(8, 8, 8, 16, 16)).mean(axis=(2, 4)).transpose(2, 0, 1).reshape(16, 64)


def normalize(tokens, config, normalized=False):
    x = np.asarray(tokens, dtype=np.float32)
    if x.shape != (16, 64) or not np.isfinite(x).all():
        raise ValueError('Expected finite tokens shaped [16,64]')
    if normalized:
        if (x < 0).any() or (x > 5).any():
            raise ValueError('Normalized tokens must be in [0,5]')
        return x
    if (x < 0).any():
        raise ValueError('Encoded RF amplitudes must be nonnegative')
    return np.clip(x / np.float32(config['gain']), 0, 5)


def quantize(x):
    return np.rint(x * INPUT_SCALE).astype('<u2')


def infer_float(p, x):
    m1 = np.zeros(8, np.float32)
    m2 = m1.copy()
    total = m1.copy()
    traces = []
    for token in x:
        v1 = np.float32(0.875)*m1 + (p['fc1.weight'] @ token + p['fc1.bias'])
        s1 = v1 >= 0.5
        m1 = v1 - np.float32(0.5)*s1
        v2 = np.float32(0.875)*m2 + (p['fc2.weight'] @ s1 + p['fc2.bias'])
        s2 = v2 >= 0.5
        m2 = v2 - np.float32(0.5)*s2
        total += v2
        traces.append((s1, s2, v1, v2))
    value = p['out.weight'] @ (total / 16) + p['out.bias'] + p['skip.weight'] @ x.mean(axis=0)
    return float(value[0]), traces


def infer_fixed(p, x):
    x = x.astype(np.int64)
    m1 = np.zeros(8, dtype=np.int64)
    m2 = m1.copy()
    total = m1.copy()
    threshold = VOLTAGE_SCALE // 2
    traces = []
    for token in x:
        c1 = (p['fc1.weight'] @ token + (p['fc1.bias'] << 12) + 256) >> 9
        v1 = ((7*m1) >> 3) + c1
        s1 = v1 >= threshold
        m1 = v1 - threshold*s1
        c2 = (p['fc2.weight'] @ s1 + p['fc2.bias']) << 3
        v2 = ((7*m2) >> 3) + c2
        s2 = v2 >= threshold
        m2 = v2 - threshold*s2
        total += v2
        traces.append((s1, s2, v1, v2))
    skip = int(p['skip.weight'][0] @ x.sum(axis=0))
    combined = int(p['out.weight'][0] @ total) + (int(p['out.bias'][0]) << 22) + (skip << 6)
    score = (combined + (1 << 16)) >> 17
    return score, traces


def demo_tokens():
    # Synthetic non-constant inputs in the learned model's normalized units.
    t, i = np.indices((16, 64))
    return (((t*197+i*59+(t*i)*17) % 4096) / 8192).astype(np.float32)
