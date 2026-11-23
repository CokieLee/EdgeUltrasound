import struct
import tempfile
import unittest

import numpy as np

from cnn_reference import infer, load_input, make_demo, pack_weights
from debug_cnn import head_from_pool, probe_models, smoke_cases, transact


class FakePort:
    """Exercise framing and short serial reads/writes without a board."""
    def __init__(self, reply):
        self.reply = bytearray(reply)
        self.sent = bytearray()

    def write(self, data):
        count = min(127, len(data))
        self.sent.extend(data[:count])
        return count

    def read(self, count):
        count = min(count, 73)
        data = bytes(self.reply[:count])
        del self.reply[:count]
        return data


class DebugTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.params, cls.cfg = make_demo(cls.folder.name)
        cls.x = load_input(cls.folder.name + '/demo_input.npy', 0)

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_known_output_models_against_network_reference(self):
        cases = list(smoke_cases(self.params, self.cfg, self.x))
        self.assertEqual(len(cases), 7)
        for name, params, cfg, independent in cases:
            with self.subTest(case=name):
                self.assertEqual(pack_weights(params).size, 15105)
                if independent is not None:
                    self.assertEqual(infer(params, cfg, self.x), independent)

    def test_one_hot_head_exposes_each_reference_pool_channel(self):
        original_score, layers = infer(self.params, self.cfg, self.x, True)
        for channel, params, cfg in probe_models(self.params, self.cfg):
            with self.subTest(channel=channel):
                self.assertEqual(infer(params, cfg, self.x), int(layers['pool'][channel]))
        self.assertEqual(head_from_pool(self.params, self.cfg, layers['pool']), original_score)
        self.assertEqual(original_score, -50)

    def test_framing_partial_io_and_mismatched_score_are_preserved(self):
        payload = self.x.tobytes()
        port = FakePort(b'\xa1\xd1' + struct.pack('<h', -73) + payload + b'\xd2')
        self.assertEqual(transact(port, self.params, self.cfg, self.x, 1, 'test'), -73)
        config = bytes(v for pair in zip(self.cfg['right_shifts'],
                                        self.cfg['bias_left_shifts']) for v in pair)
        self.assertEqual(port.sent, b'W' + config + pack_weights(self.params).tobytes()
                         + b'I' + payload)
        self.assertEqual(port.reply, b'')

    def test_replay_corruption_is_rejected_even_if_score_matches(self):
        payload = bytearray(self.x.tobytes())
        payload[4097] ^= 1
        port = FakePort(b'\xa1\xd1' + struct.pack('<h', -50) + payload + b'\xd2')
        with self.assertRaisesRegex(RuntimeError, 'Input replay mismatch at byte 4097'):
            transact(port, self.params, self.cfg, self.x, 1, 'test')

    def test_bad_ack_stops_before_input_upload(self):
        port = FakePort(b'\x00')
        with self.assertRaisesRegex(RuntimeError, 'waiting for weight acknowledgment'):
            transact(port, self.params, self.cfg, self.x, 1, 'test')
        self.assertEqual(len(port.sent), 1 + 10 + 15105 * 2)

    def test_bad_end_marker_is_rejected(self):
        port = FakePort(b'\xa1\xd1' + struct.pack('<h', -50) + self.x.tobytes() + b'\x00')
        with self.assertRaisesRegex(RuntimeError, 'Invalid final response marker'):
            transact(port, self.params, self.cfg, self.x, 1, 'test')


if __name__ == '__main__':
    unittest.main()
