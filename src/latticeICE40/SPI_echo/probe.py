"""Read the raw onboard SPI echo without requiring the SNN protocol."""
import argparse
import struct
from time import sleep

from ftdi_spi import EXCHANGE, FtdiSpi, SEND_IMMEDIATE

PATTERN = bytes.fromhex('a5 69 3c f0 00 ff 55 96')
INVERTED = bytes(value ^ 0xff for value in PATTERN)


def check(link):
    print(f'FTDI GPIO: 0x{link._gpio():02x} (CDONE and flash CS should both be high)')
    command = struct.pack('<BH', EXCHANGE, len(PATTERN)-1) + PATTERN + bytes([SEND_IMMEDIATE])
    print(f'Sent: {PATTERN.hex(" ")}')
    results = []
    for _ in range(3):
        link._write(command)
        result = bytes(link._read(len(PATTERN)))
        results.append(result)
        print(f'Read: {result.hex(" ")}')
        sleep(0.05)
    if all(result == INVERTED for result in results):
        print('PASS: SPI data path works and GPIO 20 saw clock edges.')
        return 0
    if all(result == PATTERN for result in results):
        print('SPI data path works. GPIO 20 did not see enough clock edges. Check 12M to 20 or OSC/R16.')
        return 2
    print('SPI echo failed. Check the flashed probe image and the onboard flash wiring.')
    return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ftdi', default='ftdi://ftdi:232h/1')
    args = parser.parse_args()
    with FtdiSpi(args.ftdi) as link:
        raise SystemExit(check(link))
