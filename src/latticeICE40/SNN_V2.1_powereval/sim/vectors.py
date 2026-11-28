"""Generate complete wire-protocol vectors from the fixed-point reference."""
from binascii import crc_hqx
from pathlib import Path
import struct
import sys

import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root/'host'))
from reference import demo_tokens, infer_fixed, load_weights, quantize


def test_frames():
    x = demo_tokens()
    t, i = np.indices(x.shape)
    stress = np.where((i+t)%3 == 0, 5, np.where(t%2==0, 0, x))
    return (quantize(x), quantize(x), quantize(stress.astype(np.float32)))


def reply(q, fixed, trace):
    score, steps = infer_fixed(fixed, q)
    output = bytearray()
    for t, (s1, s2, v1, v2) in enumerate(steps):
        if trace:
            output.extend((0xc0+t, int(np.packbits(s1, bitorder='little')[0]),
                           int(np.packbits(s2, bitorder='little')[0])))
            output.extend(v1.astype('<i4').tobytes())
            output.extend(v2.astype('<i4').tobytes())
        if t < 15:
            output.append(0xa1)
    output.extend(b'\xd1'+struct.pack('<iH', score, crc_hqx(q.tobytes(), 0xffff)))
    return bytes(output)


if __name__ == '__main__':
    cfg, _, fixed = load_weights(root/'weights')
    frames = test_frames()
    stimulus = b''.join(q.tobytes() for q in frames)
    response = b'\xb7\x01'+bytes.fromhex(cfg['weights_sha256'][:16])
    response += b''.join(reply(q, fixed, trace) for q, trace in zip(frames, (True,False,True)))
    folder = root/'build/vectors'
    folder.mkdir(parents=True, exist_ok=True)
    (folder/'input.mem').write_text(''.join(f'{b:02x}\n' for b in stimulus))
    (folder/'output.mem').write_text(''.join(f'{b:02x}\n' for b in response))
    print(f'Expected {len(response)} protocol bytes for three frames. Demo score {infer_fixed(fixed, frames[0])[0]}')
