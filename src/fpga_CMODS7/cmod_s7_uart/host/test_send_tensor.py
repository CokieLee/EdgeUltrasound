import tempfile
import unittest
from pathlib import Path

from send_tensor import payload_from_file, transfer


class EchoPort:
    def __init__(self, corrupt=False, drop=False):
        self.queue = bytearray()
        self.sent = bytearray()
        self.corrupt = corrupt
        self.drop = drop

    def write(self, data):
        # Exercise partial serial writes, not just the happy path.
        block = bytes(data[:17])
        self.sent.extend(block)
        if not self.drop:
            self.queue.extend(block)
        return len(block)

    def read(self, size):
        result = bytes(self.queue[:min(size, 11)])
        del self.queue[:len(result)]
        if self.corrupt and result:
            self.corrupt = False
            result = bytes([result[0] ^ 1]) + result[1:]
        return result


class TransferTests(unittest.TestCase):
    def test_all_bytes_and_final_short_block(self):
        payload = bytes(range(256))*4 + b'last'
        port = EchoPort()
        self.assertEqual(transfer(port, payload), len(payload))
        self.assertEqual(bytes(port.sent), payload)

    def test_corruption_is_not_reported_as_success(self):
        with self.assertRaisesRegex(ValueError, 'byte 0'):
            transfer(EchoPort(corrupt=True), b'abc')

    def test_missing_echo_times_out(self):
        with self.assertRaises(TimeoutError):
            transfer(EchoPort(drop=True), b'abc', timeout=0.002)

    def test_chunk_limit(self):
        with self.assertRaises(ValueError):
            transfer(EchoPort(), b'abc', chunk_size=257)

    def test_signed_and_float_wire_formats(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'input.npy'
            values = np.array([[[[-32768, -1, 0, 32767]]]], dtype='>i2')
            np.save(path, values)
            payload, shape, _ = payload_from_file(path)
            self.assertEqual(shape, (1, 1, 1, 4))
            self.assertEqual(payload, b'\x00\x80\xff\xff\x00\x00\xff\x7f')
            values = np.array([0.0, -1.5, 2.25], dtype=np.float32)
            np.save(path, values)
            self.assertEqual(payload_from_file(path)[0], values.astype('<f4').tobytes())

    def test_rejects_implicit_float64_conversion(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'input.npy'
            np.save(path, np.array([1.0], dtype=np.float64))
            with self.assertRaises(ValueError):
                payload_from_file(path)


if __name__ == '__main__':
    unittest.main()
