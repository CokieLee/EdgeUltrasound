"""Check the checkpoint export, RF layout, arithmetic and UART host protocol."""
import hashlib
from binascii import crc_hqx
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np

from vectors import reply, root, test_frames
from export_weights import export, KEYS
from reference import (EVENTS, SCORE_SCALE, demo_tokens, encode_rf, infer_fixed,
                       infer_float, load_weights, normalize, quantize)
from run_snn_v2 import exchange


class Loopback(io.BytesIO):
    """Keep the FPGA's given response separate from bytes sent by the host."""
    def __init__(self, response):
        super().__init__(response)
        self.transmitted = bytearray()

    def write(self, payload):
        self.transmitted.extend(payload)
        return len(payload)


class HostTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg, cls.floating, cls.fixed = load_weights(root/'weights')
        cls.x = demo_tokens()

    def test_checkpoint_rom(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            cfg = export(root/'weights/source.pt', path)
            self.assertEqual(cfg, self.cfg)
            for name in ('lo.mem','hi.mem','model_id.svh'):
                self.assertEqual((path/name).read_bytes(), (root/'weights'/name).read_bytes())
            low = bytes.fromhex((path/'lo.mem').read_text())
            high = bytes.fromhex((path/'hi.mem').read_text())
            word_bytes = bytes(b for word in zip(low,high) for b in word)
            self.assertEqual(len(word_bytes), 1330)
            self.assertEqual(hashlib.sha256(word_bytes).hexdigest(), cfg['weights_sha256'])
            np.testing.assert_array_equal(np.frombuffer(word_bytes, dtype='<i2'),
                                          np.concatenate([self.fixed[k].ravel() for k in KEYS]))

    def test_rf_spatial_and_temporal_layout(self):
        rf = np.zeros((384,64,250), dtype=np.int16)
        rf[EVENTS[6],40:48,48:64] = -128
        rf[EVENTS[2],56:64,240:250] = -80
        tokens = encode_rf(rf)
        self.assertEqual(tokens[3,6*8+5], 128)
        self.assertEqual(tokens[15,2*8+7], 50)
        self.assertEqual(np.count_nonzero(tokens), 2)
        self.assertAlmostEqual(float(normalize(tokens,self.cfg)[3,53]), 128/self.cfg['gain'], places=6)

    def test_fixed_demo_against_float(self):
        fixed, fixed_trace = infer_fixed(self.fixed, quantize(self.x))
        floating, float_trace = infer_float(self.floating, self.x)
        self.assertEqual(fixed, -1055882)
        self.assertLess(abs(fixed/SCORE_SCALE-floating), 0.0001)
        for a, b in zip(fixed_trace, float_trace):
            for fixed_spikes, float_spikes in zip(a[:2],b[:2]):
                np.testing.assert_array_equal(fixed_spikes, float_spikes)
        self.assertGreater(sum(row[0].sum() for row in fixed_trace), 0)
        self.assertGreater(sum(row[1].sum() for row in fixed_trace), 0)

    def test_uart_frames(self):
        identity = b'\xb7\x01'+bytes.fromhex(self.cfg['weights_sha256'][:16])
        for q, with_trace in zip(test_frames(), (True,False,True)):
            score, steps = infer_fixed(self.fixed,q)
            port = Loopback(identity+reply(q,self.fixed,with_trace))
            actual, crc, trace = exchange(port,q,self.cfg,steps,with_trace)
            self.assertEqual(actual, score)
            self.assertEqual(crc, crc_hqx(q.tobytes(), 0xffff))
            self.assertEqual(port.transmitted, b'H'+(b'T' if with_trace else b'I')+q.tobytes())
            self.assertEqual(len(trace),16 if with_trace else 0)
            self.assertTrue(all(step['match'] for step in trace))
            self.assertEqual(port.read(),b'')
        with self.assertRaisesRegex(RuntimeError, 'Wrong FPGA bitstream'):
            exchange(Loopback(b'\0'*10),q,self.cfg,steps,False)


if __name__ == '__main__':
    unittest.main()
