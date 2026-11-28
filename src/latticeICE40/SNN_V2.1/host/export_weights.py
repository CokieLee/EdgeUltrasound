"""Convert the supplied CPU PyTorch checkpoint into FPGA ROMs and a NumPy reference."""
import argparse
from collections import OrderedDict
import hashlib
import io
import json
from pathlib import Path
import pickle
import re
from zipfile import ZipFile

import numpy as np

WEIGHT_FRACTION = 15
KEYS = {
    'fc1.weight': (8, 64), 'fc1.bias': (8,), 'fc2.weight': (8, 8),
    'fc2.bias': (8,), 'out.weight': (1, 8), 'out.bias': (1,),
    'skip.weight': (1, 64),
}


class FloatCheckpoint(pickle.Unpickler):
    """Read the specific ZIP/FloatStorage checkpoint format without importing torch."""
    def __init__(self, raw_pickle, archive, prefix):
        super().__init__(io.BytesIO(raw_pickle))
        self.archive, self.prefix = archive, prefix

    def find_class(self, module, name):
        allowed = {('collections', 'OrderedDict'): OrderedDict,
                   ('torch', 'FloatStorage'): 'float32',
                   ('torch._utils', '_rebuild_tensor_v2'): self.tensor}
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f'Unsupported checkpoint global {module}.{name}')
        return allowed[module, name]

    def persistent_load(self, entry):
        kind, dtype, key, device, size = entry
        if (kind, dtype, device) != ('storage', 'float32', 'cpu') or not re.fullmatch(r'[0-9]+', key):
            raise pickle.UnpicklingError('Expected numbered CPU float storage')
        info = self.archive.getinfo(self.prefix + 'data/' + key)
        if not 0 <= size <= 4096 or info.file_size != size*4:
            raise ValueError('Unexpected storage length')
        return np.frombuffer(self.archive.read(info), dtype='<f4')

    @staticmethod
    def tensor(storage, offset, shape, stride, requires_grad, hooks):
        if (not 1 <= len(shape) <= 2 or any(type(n) is not int or n <= 0 for n in shape)
                or tuple(stride) != ((shape[1], 1) if len(shape) == 2 else (1,))):
            raise ValueError('Expected contiguous tensor')
        length = int(np.prod(shape))
        if not 0 <= offset <= storage.size - length:
            raise ValueError('Invalid storage range')
        return storage[offset:offset+length].reshape(shape).copy()


def read_checkpoint(path):
    with ZipFile(path) as archive:
        item, = (entry for entry in archive.infolist() if entry.filename.endswith('/data.pkl'))
        prefix = item.filename[:-len('data.pkl')]
        if item.file_size > 8192 or archive.read(prefix+'byteorder') != b'little':
            raise ValueError('Expected a small little-endian checkpoint')
        data = FloatCheckpoint(archive.read(item), archive, prefix).load()
    state = data['state_dict']
    if set(state) != KEYS.keys():
        raise ValueError('Checkpoint must be an 8-neuron model with the skip branch')
    for key, shape in KEYS.items():
        if state[key].shape != shape or not np.isfinite(state[key]).all():
            raise ValueError(f'Invalid {key}')
    return data


def export(source, folder, config_path=None):
    data = read_checkpoint(source)
    state = data['state_dict']
    gain, target_mean = float(data['gain']), float(data['target_mean'])
    if not np.isfinite([gain, target_mean]).all() or gain <= 0:
        raise ValueError('Invalid gain or target mean')
    if data.get('model', 'snn_skip8') != 'snn_skip8' or data.get('architecture', 'skip8') != 'skip8':
        raise ValueError('Only the spiking skip8 checkpoint is supported')
    if config_path is not None:
        supplied = json.loads(config_path.read_text(encoding='utf-8-sig'))
        expected = dict(model='snn_skip8', gain=gain, target_mean=target_mean,
                        clip=5.0, beta=0.875, threshold=0.5, hidden_width=8, steps=16,
                        features=64, parameters=665, target_scale=10, seed=data['seed'])
        for key in ('train_rats', 'validation_rats'):
            if key in data:
                expected[key] = data[key]
        for key, value in expected.items():
            if key in supplied and supplied[key] != value:
                raise ValueError(f'{key} mismatch: config {supplied[key]!r}, expected {value!r}')
    quantized = {}
    for name, values in state.items():
        scaled = np.rint(values.astype(np.float64) * (1 << WEIGHT_FRACTION))
        if np.any((scaled < -32768) | (scaled > 32767)):
            raise ValueError(f'{name} exceeds the signed 16-bit weight format')
        quantized[name] = scaled.astype('<i2')
    words = np.concatenate([quantized[key].ravel() for key in KEYS]).astype('<i2')
    identity = hashlib.sha256(words.tobytes()).hexdigest()
    folder.mkdir(parents=True, exist_ok=True)
    (folder/'lo.mem').write_text(''.join(f'{int(v)&255:02x}\n' for v in words), encoding='ascii')
    (folder/'hi.mem').write_text(''.join(f'{(int(v)>>8)&255:02x}\n' for v in words), encoding='ascii')
    (folder/'model_id.svh').write_text(f"localparam [63:0] MODEL_ID = 64'h{identity[:16]};\n")
    np.savez(folder/'reference.npz', **state, **{'q_'+k: v for k,v in quantized.items()})
    config = dict(architecture='skip8', width=8, skip=True, gain=gain,
                  target_mean=target_mean, target_scale=10, seed=data['seed'],
                  checkpoint_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  weights_sha256=identity, parameters=int(words.size),
                  weight_fraction=WEIGHT_FRACTION, input_fraction=12, voltage_fraction=18,
                  score_fraction=20)
    if 'model' in data:
        config.update(model='snn_skip8', clip=5.0)
    for key in ('train_rats', 'validation_rats'):
        if key in data:
            config[key] = data[key]
    (folder/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--config', type=Path, help='Check the source model config against the checkpoint and RTL')
    args = parser.parse_args()
    print(json.dumps(export(args.checkpoint, args.output, args.config), indent=2))
