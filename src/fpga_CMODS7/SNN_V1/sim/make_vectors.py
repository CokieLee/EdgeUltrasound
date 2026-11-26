"""Generate bit-exact vectors for the RTL and UART simulations."""
from pathlib import Path
import struct
import sys

import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root/'host'))
from snn_reference import infer, load_snn_model
from tensor_ops import load_input, pack_weights

output = root/'build/vectors'
output.mkdir(parents=True, exist_ok=True)
demo = root/'demo'
params, cfg = load_snn_model(demo/'demo_weights.npz', demo/'demo_config.json')
x = load_input(demo/'demo_input.npy', 0)
score, trace = infer(params, cfg, x)
weights = pack_weights(params)
config = bytes(v for pair in zip(cfg['right_shifts'], cfg['bias_left_shifts']) for v in pair)


def hex_file(name, values, digits):
    mask = (1 << (4*digits)) - 1
    (output/name).write_text(''.join(f'{int(v)&mask:0{digits}x}\n' for v in values))


hex_file('weights.mem', weights, 4)
hex_file('input.mem', x.ravel(), 4)
hex_file('config.mem', config, 2)
hex_file('score.mem', [score], 4)
hex_file('trace.mem', np.concatenate([layer.ravel() for layer in trace]), 4)
hex_file('uart_stimulus.mem', b'W'+config+weights.tobytes()+b'T'+x.tobytes(), 2)
stream = b''.join(bytes([0xc0+i])+layer.astype('<i2').tobytes() for i, layer in enumerate(trace))
reply = b'\xd1'+struct.pack('<h', score)+x.tobytes()+b'\xd2'
(output/'expected_uart.bin').write_bytes(b'\xb6\x01\xa1'+stream+reply)
print(f'Reference score {score}; UART response {len(stream)+len(reply)+3} bytes')
