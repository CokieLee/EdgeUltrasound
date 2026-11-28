"""Run the complete host exchange against mocked FTDI slots."""
import io
from contextlib import redirect_stdout
from binascii import crc_hqx
import struct
from types import SimpleNamespace
from unittest.mock import patch

from vectors import reply, root, test_frames
from ftdi_spi import FtdiSpi
from reference import infer_fixed, load_weights
from run_snn_v2 import exchange


class MockFtdi:
    def __init__(self, identity, response):
        self.identity = identity
        self.response = response
        self.flash_cs = 1
        self.data = b''
        self.received = bytearray()

    def open_mpsse_from_url(self, url, **kwargs):
        assert kwargs['direction'] == 0

    def close(self):
        pass

    def write_data(self, command):
        if command[0] == 0x80:
            self.flash_cs = int(not bool(command[2] & 0x10))
        elif command == b'\x81\x87':
            self.data = bytes([0x40 | (self.flash_cs << 4)])
        else:
            assert self.flash_cs == 1
            assert command[0]==0x31 and command[-1]==0x87
            size = struct.unpack_from('<H', command,1)[0]+1
            slots = command[3:-1]
            assert size==len(slots)
            sent = bytearray()
            for tag,value in zip(slots[::2],slots[1::2]):
                sent += b'\x01'+self.pending[:1] if self.pending else b'\x00\x00'
                self.pending = self.pending[1:]
                if tag==1:
                    self.received.append(value)
                    n=len(self.received)
                    if n==1:
                        self.pending=self.identity
                    elif n>2 and (n-2)%128==0:
                        step=(n-2)//128
                        size=68 if step<16 else 74
                        self.pending,self.response=self.response[:size],self.response[size:]
            self.data=bytes(sent)
        return len(command)

    pending = b''

    def read_data_bytes(self, count, attempt):
        assert count==len(self.data)
        result,self.data=self.data,b''
        return result


if __name__ == '__main__':
    cfg, _, fixed = load_weights(root/'weights')
    identity = b'\xb7\x01'+bytes.fromhex(cfg['weights_sha256'][:16])
    for q in test_frames():
        mock = MockFtdi(identity,reply(q,fixed,True))
        with patch.dict('sys.modules', {'pyftdi.ftdi': SimpleNamespace(Ftdi=lambda: mock)}):
            score, trace = infer_fixed(fixed,q)
            with FtdiSpi('mock') as port, redirect_stdout(io.StringIO()):
                actual, crc, steps = exchange(port,q,cfg,trace,True)
            assert actual==score and crc==crc_hqx(q.tobytes(),0xffff)
            assert mock.received==b'HT'+q.tobytes()
            assert len(steps)==16 and all(step['match'] for step in steps)
    print('PASS FTDI handshake, batched slots, Q20, CRC and spike/voltage traces')
