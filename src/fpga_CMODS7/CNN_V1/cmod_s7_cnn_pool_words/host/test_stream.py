import contextlib
import io
import struct
import tempfile
import unittest
from pathlib import Path
import numpy as np

from cnn_reference import LAYERS, conv, infer, load_input, make_demo, pack_weights, requantize
from run_stream import STAGES, CapturePort, actual_pool, collect, comparison, passed


class ShortPort(CapturePort):
    def __init__(self, data):
        super().__init__(io.BytesIO(data))
        self.sent = bytearray()

    def read(self, count):
        return super().read(min(count, 107))

    def write(self, data):
        count = min(len(data), 109)
        self.sent.extend(data[:count])
        return count


class StreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.params, cls.cfg = make_demo(cls.folder.name)
        cls.x = load_input(Path(cls.folder.name) / 'demo_input.npy', 0)

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def capture(self, fault=None, streaming=True, with_sums=False):
        previous, layers = self.x, {}
        for i, (name, _) in enumerate(LAYERS[:4]):
            result = conv(previous, self.params[name + '.weight'], self.params[name + '.bias'],
                          1 if i == 0 else 2, 0 if i == 0 else 1,
                          self.cfg['right_shifts'][i], self.cfg['bias_left_shifts'][i], i != 0)
            if name == 'conv1' and fault == 'conv1':
                result[2, 3, 4] += 25
            layers[name] = previous = result
        sums = previous.astype(np.int64).sum(axis=(1, 2))
        pooled = actual_pool(previous)
        if fault == 'sum':
            sums[0] = -161
            pooled[0] = -2
        if fault == 'pool':
            pooled[0] += 1
        layers['pool'] = pooled
        acc = self.params['head.weight'].astype(np.int64) @ pooled.astype(np.int64)
        acc += self.params['head.bias'].astype(np.int64)
        score = int(requantize(acc, 0, False)[0])
        data = b'\xb5' + bytes([2 if with_sums else 1]) + b'\xa1'
        if streaming:
            for i, (name, _) in enumerate(STAGES):
                data += bytes([0xc0 + i])
                if name == 'pool' and with_sums:
                    data += b''.join(struct.pack('<h', int(value)) +
                                     int(total).to_bytes(6, 'little', signed=True)
                                     for value, total in zip(pooled, sums))
                else:
                    data += layers[name].astype('<i2').tobytes()
        data += b'\xd1' + struct.pack('<h', score) + self.x.tobytes() + b'\xd2'
        return data

    def analyze(self, capture, streaming=True):
        port = ShortPort(capture)
        with tempfile.TemporaryDirectory() as output:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                report = collect(port, self.params, self.cfg, self.x, output, 1, streaming)
            saved = [p.name for p in Path(output).iterdir()]
        return report, port, saved

    def test_complete_stream_and_exact_original_upload_layout(self):
        report, port, saved = self.analyze(self.capture())
        self.assertTrue(passed(report))
        self.assertEqual(report['score']['actual'], -50)
        self.assertFalse(report['parameter_readback_verified'])
        self.assertFalse(report['layer_transport_crc'])
        config = bytes(v for pair in zip(self.cfg['right_shifts'], self.cfg['bias_left_shifts']) for v in pair)
        self.assertEqual(port.sent, b'HW' + config + pack_weights(self.params).tobytes() + b'T' + self.x.tobytes())
        self.assertIn('stem.npy', saved)
        self.assertIn('pool.npy', saved)

    def test_continuous_mode_uses_original_inference_response(self):
        report, port, _ = self.analyze(self.capture(streaming=False), streaming=False)
        self.assertTrue(passed(report))
        self.assertEqual(report['layers'], [])
        self.assertTrue(port.sent.endswith(b'I' + self.x.tobytes()))

    def test_local_checks_distinguish_propagation_from_new_fault(self):
        report, _, _ = self.analyze(self.capture('conv1'))
        self.assertTrue(report['completed'])
        self.assertFalse(passed(report))
        self.assertEqual(report['first_reference_mismatch'], 'conv1')
        self.assertEqual(report['first_local_mismatch'], 'conv1')
        for entry in report['layers'][2:]:
            self.assertTrue(entry['given_emitted_input_and_sent_parameters']['match'])

    def test_pool_fault_does_not_blame_consistent_head(self):
        report, _, _ = self.analyze(self.capture('pool'))
        self.assertEqual(report['first_local_mismatch'], 'pool')
        self.assertTrue(report['score']['local_match'])

    def test_raw_pool_sums_separate_sum_and_quotient_errors(self):
        for fault, expected_matches in ((None, (True, True)),
                                        ('sum', (False, True)),
                                        ('pool', (True, False))):
            with self.subTest(fault=fault):
                report, _, saved = self.analyze(self.capture(fault, with_sums=True))
                self.assertTrue(report['completed'])
                self.assertIn('pool_sums.npy', saved)
                diagnostic = report['pool_diagnostic']
                matches = (diagnostic['sum_vs_emitted_conv3']['match'],
                           diagnostic['quotient_vs_fpga_sum_divided_by_80']['match'])
                self.assertEqual(matches, expected_matches)
                self.assertEqual(diagnostic['channels'][0]['expected_sum'], 7581)
                if fault == 'sum':
                    self.assertEqual(diagnostic['channels'][0]['fpga_sum'], -161)

    def test_layer_tag_failure_aborts(self):
        data = bytearray(self.capture())
        data[3] = 0xcf
        report, _, saved = self.analyze(data)
        self.assertFalse(report['completed'])
        self.assertIn('Unexpected layer tag', report['error'])
        self.assertIn('report.json', saved)

    def test_truncation_and_invalid_footer_cannot_pass(self):
        for data in (self.capture()[:-100], self.capture()[:-1] + b'X'):
            with self.subTest(length=len(data)):
                report, _, _ = self.analyze(data)
                self.assertFalse(report['completed'])
                self.assertFalse(passed(report))

    def test_replay_corruption_is_reported(self):
        data = bytearray(self.capture())
        data[-20] ^= 1
        report, _, _ = self.analyze(data)
        self.assertTrue(report['completed'])
        self.assertFalse(report['input_replay_match'])
        self.assertFalse(passed(report))

    def test_signed_error_and_invalid_negative_pool_input(self):
        result = comparison(np.array([-32768], '<i2'), np.array([32767], '<i2'))
        self.assertEqual(result['max_abs_error'], 65535)
        x = np.zeros((32, 4, 20), dtype=np.int16)
        x[0, 0, 0] = -1
        self.assertEqual(int(actual_pool(x)[0]), 0)


if __name__ == '__main__':
    unittest.main()
