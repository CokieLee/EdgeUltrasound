from pathlib import Path
import struct
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'host'))
from cnn_reference import infer, load_input, make_demo, pack_weights
from run_stream import STAGES

folder = ROOT / 'build' / 'vectors'
folder.mkdir(parents=True, exist_ok=True)
params, cfg = make_demo(folder)
x = load_input(folder / 'demo_input.npy', 0)
score, layers = infer(params, cfg, x, True)


def save(name, array):
    (folder / (name + '.mem')).write_text(''.join(
        f'{int(v) & 65535:04x}\n' for v in np.asarray(array).reshape(-1)))


save('weights', pack_weights(params))
save('input', x)
save('score', [score])
for name, values in layers.items():
    save(name, values)
config = bytes(v for pair in zip(cfg['right_shifts'], cfg['bias_left_shifts']) for v in pair)
stimulus = b'W' + config + pack_weights(params).tobytes() + b'T' + x.tobytes()
assert len(stimulus) == 40206
(folder / 'uart_stimulus.mem').write_text(''.join(f'{v:02x}\n' for v in stimulus))
expected = b'\xb5\x01\xa1'
for i, (name, _) in enumerate(STAGES):
    expected += bytes([0xc0 + i]) + layers[name].astype('<i2').tobytes()
expected += b'\xd1' + struct.pack('<h', score) + x.tobytes() + b'\xd2'
assert len(expected) == 154956
(folder / 'reference_capture.bin').write_bytes(expected)
print(f'Generated reference vectors: score {score}, {len(expected)} device response bytes.')
