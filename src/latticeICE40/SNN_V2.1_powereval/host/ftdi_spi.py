"""FT232H MPSSE transport for the stock UPduino flash wiring."""
import struct
from time import sleep

SET_LOW = 0x80
GET_LOW = 0x81
SEND_IMMEDIATE = 0x87
# Full duplex, MSB first, drive on falling edge, sample on rising edge.
EXCHANGE = 0x31
CLOCK_DATA_OUTPUTS = 0x03
FLASH_CS = 0x10
CDONE = 0x40


class FtdiSpi:
    def __init__(self, url):
        self.url = url
        self.device = None
        self.pending = bytearray()

    def __enter__(self):
        from pyftdi.ftdi import Ftdi

        self.device = Ftdi()
        try:
            # Begin with all pins as inputs so the FPGA can finish flash boot.
            self.device.open_mpsse_from_url(self.url, direction=0, initial=0, frequency=250_000)
            self._check_boot()
            # Hold SCK low. Briefly select flash without clocking it to reset
            # the FPGA's application protocol, then release CS to its pull-up.
            self._write(bytes([SET_LOW, 0, CLOCK_DATA_OUTPUTS | FLASH_CS]))
            if self._gpio() & FLASH_CS:
                raise OSError('Could not assert flash select for protocol reset')
            self._write(bytes([SET_LOW, 0, CLOCK_DATA_OUTPUTS]))
            self._check_boot()
            return self
        except Exception:
            self.device.close()
            raise

    def __exit__(self, *_):
        self.device.close()

    def _check_boot(self):
        for _ in range(20):
            if (self._gpio() & (FLASH_CS | CDONE)) == (FLASH_CS | CDONE):
                return
            sleep(0.025)
        raise OSError('UPduino has not booted, or flash chip select is low')

    def _gpio(self):
        self._write(bytes([GET_LOW, SEND_IMMEDIATE]))
        return self._read(1)[0]

    def _write(self, payload):
        if self.device.write_data(payload) != len(payload):
            raise OSError('Incomplete FTDI command')

    def _read(self, count):
        reply = self.device.read_data_bytes(count, attempt=8)
        if len(reply) != count:
            raise OSError(f'FTDI returned {len(reply)} of {count} SPI bytes')
        return reply

    def _exchange(self, slots):
        self._write(struct.pack('<BH', EXCHANGE, len(slots)-1) + slots + bytes([SEND_IMMEDIATE]))
        reply = self._read(len(slots))
        for tag, value in zip(reply[::2], reply[1::2]):
            if tag == 1:
                self.pending.append(value)
            elif tag != 0:
                raise OSError(f'Invalid SPI slot flag 0x{tag:02x}; check flash image and 12 MHz clock')

    def write(self, payload):
        if payload:
            slots = bytearray(2*len(payload))
            slots[::2] = b'\x01'*len(payload)
            slots[1::2] = payload
            self._exchange(slots)
        return len(payload)

    def read(self, count):
        if len(self.pending) < count:
            self._exchange(bytes(2*(count-len(self.pending))))
        data = bytes(self.pending[:count])
        del self.pending[:count]
        return data
