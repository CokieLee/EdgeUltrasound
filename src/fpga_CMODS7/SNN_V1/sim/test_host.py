"""Host contract checks without a board or HDL simulator."""
import io
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root/'host'))
from export_float import export
from run_snn import collect
from snn_reference import infer, lif, load_snn_model, score_from_counts
from tensor_ops import load_input


class HostTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        demo = root/'demo'
        cls.params, cls.cfg = load_snn_model(demo/'demo_weights.npz', demo/'demo_config.json')
        cls.x = load_input(demo/'demo_input.npy', 0)
        cls.expected = infer(cls.params, cls.cfg, cls.x)

    def test_delayed_reset_and_exact_threshold(self):
        previous = np.array([0, 0, 1024, 1025, -1], dtype=np.int64)
        current = np.array([1024, 1025, 103, 0, 0], dtype=np.int16)
        spike, membrane = lif(previous, current)
        np.testing.assert_array_equal(membrane, [1024, 1025, 1024, -102, -1])
        np.testing.assert_array_equal(spike, [0, 1, 0, 0, 0])

    def test_average_and_signed_floor(self):
        expected, layers = self.expected
        actual = [
            self.params['head.weight'][0].astype(np.int64)
            @ ((layers[3] >> t) & 1).sum(axis=(1, 2)) / 80
            + int(self.params['head.bias'][0])
            for t in range(4)
        ]
        self.assertEqual(int(np.floor(np.mean(actual))), expected)
        params = {'head.weight': -np.ones((1, 32), dtype=np.int16),
                  'head.bias': np.zeros(1, dtype=np.int16)}
        self.assertEqual(score_from_counts(params, self.cfg, np.ones(32)), -1)

    def test_report_parser(self):
        expected, layers = self.expected
        capture = b''.join(bytes([0xc0+i])+layer.astype('<i2').tobytes()
                           for i, layer in enumerate(layers))
        reply = b'\xd1'+struct.pack('<h', expected)+self.x.tobytes()+b'\xd2'
        for continuous in (False, True):
            source = io.BytesIO(reply if continuous else capture+reply)
            result = collect(source, self.expected, self.x, continuous, 1, self.params, self.cfg)
            self.assertTrue(result['match'])
            self.assertEqual(source.read(), b'')
        with self.assertRaisesRegex(RuntimeError, 'stem tag'):
            collect(io.BytesIO(b'\x00'+capture[1:]+reply), self.expected, self.x,
                    False, 1, self.params, self.cfg)

    def test_export_round_trip(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            float_weights = {}
            for key, values in self.params.items():
                float_weights[key] = values.astype(np.float32) / 1024
            np.savez(folder/'float.npz', **float_weights)
            np.save(folder/'float.npy', self.x.astype(np.float32)/1024)
            export(folder/'float.npz', folder/'float.npy', folder/'exported', 0)
            out = folder/'exported'
            params, cfg = load_snn_model(out/'weights.npz', out/'config.json')
            score, actual = infer(params, cfg, load_input(out/'input.npy', 0))
            counts = self.expected[1][-1].astype(np.int64)
            weights = self.params['head.weight'][0].astype(np.int64)
            exact = (float(weights @ counts) / 320 + int(self.params['head.bias'][0])) / 1024
            lsb = 2 ** -cfg['score_fractional_bits']
            self.assertEqual(score, int(np.floor(exact / lsb)))
            for got, reference in zip(actual, self.expected[1]):
                np.testing.assert_array_equal(got, reference)


if __name__ == '__main__':
    unittest.main()
