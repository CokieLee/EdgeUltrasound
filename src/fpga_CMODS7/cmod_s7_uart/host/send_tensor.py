#!/usr/bin/env python3
"""Send .npy tensor bytes over UART and verify every echoed byte.

No quantization, header, tensor parsing, or inference is performed by the FPGA.
The host knows the shape/dtype; the FPGA initially sees only a byte stream.
"""
import argparse
import hashlib
import time
from pathlib import Path


def payload_from_file(path):
    import numpy as np
    tensor = np.load(path, allow_pickle=False)
    if not isinstance(tensor, np.ndarray):
        raise ValueError("Use a single .npy array, not .npz or a torch checkpoint.")
    if (tensor.dtype.kind, tensor.dtype.itemsize) not in {
        ('f', 4), ('i', 2), ('i', 1), ('u', 1)
    }:
        raise ValueError("Supported types: float32, int16, int8, uint8; no automatic quantization.")
    wire = np.ascontiguousarray(tensor, dtype=tensor.dtype.newbyteorder('<'))
    return wire.tobytes(order='C'), tuple(wire.shape), str(wire.dtype)


def write_all(port, data):
    offset = 0
    while offset < len(data):
        count = port.write(data[offset:])
        if count is None or count <= 0:
            raise TimeoutError("UART write made no progress")
        offset += count


def read_exact(port, size, timeout):
    result = bytearray()
    deadline = time.monotonic() + timeout
    while len(result) < size:
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Expected {size} echo bytes, received {len(result)}")
        result.extend(port.read(size - len(result)))
    return bytes(result)


def transfer(port, payload, chunk_size=256, timeout=3.0, progress=None):
    if not 1 <= chunk_size <= 256:
        raise ValueError("Chunk size must be between 1 and 256 bytes")
    if timeout <= 0:
        raise ValueError("Timeout must be positive")
    for offset in range(0, len(payload), chunk_size):
        block = payload[offset:offset + chunk_size]
        write_all(port, block)
        echo = read_exact(port, len(block), timeout)
        if echo != block:
            first = next(i for i, (a, b) in enumerate(zip(block, echo)) if a != b)
            raise ValueError(f"Echo mismatch at byte {offset + first}: "
                             f"sent 0x{block[first]:02x}, got 0x{echo[first]:02x}")
        if progress:
            progress(offset + len(block), len(payload))
    return len(payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list-ports', action='store_true')
    parser.add_argument('--port', help='For example COM5 on Windows, /dev/ttyUSB1 in WSL')
    parser.add_argument('--baud', type=int, default=115200)
    parser.add_argument('--file', type=Path, help='.npy tensor; otherwise send a 1024-byte link test')
    parser.add_argument('--chunk-size', type=int, default=256)
    parser.add_argument('--timeout', type=float, default=3.0)
    args = parser.parse_args()
    try:
        import serial
        from serial.tools import list_ports
    except ImportError:
        parser.exit(1, 'Install dependencies: python -m pip install pyserial numpy\n')
    if args.list_ports:
        for entry in list_ports.comports():
            print(f'{entry.device}: {entry.description} [{entry.hwid}]')
        return
    if not args.port:
        parser.error('--port is required (or use --list-ports)')
    if args.baud <= 0 or args.timeout <= 0 or not 1 <= args.chunk_size <= 256:
        parser.error('Positive baud/timeout and chunk-size 1..256 are required')
    if args.file:
        payload, shape, dtype = payload_from_file(args.file)
        print(f'Shape: {shape}; wire dtype: {dtype}; order: C (last axis varies fastest)')
    else:
        payload = bytes(range(256)) * 4
        print('Link test: 1024 bytes covering all byte values')
    if not payload:
        parser.error('Empty input is not a useful link test')
    print(f'Bytes: {len(payload):,}; SHA-256: {hashlib.sha256(payload).hexdigest()}')
    print(f'Ideal one-direction wire time: {len(payload)*10/args.baud:.2f} s; '
          'block echo checking adds overhead.')
    last_print = [0.0]

    def progress(done, total):
        now = time.monotonic()
        if now - last_print[0] >= 1 or done == total:
            print(f'\rVerified {done:,}/{total:,} bytes ({100*done/total:.1f}%)', end='', flush=True)
            last_print[0] = now

    try:
        with serial.Serial(args.port, args.baud, timeout=0.05, write_timeout=args.timeout,
                           bytesize=8, parity='N', stopbits=1,
                           xonxoff=False, rtscts=False, dsrdtr=False) as port:
            port.reset_input_buffer()
            started = time.monotonic()
            transfer(port, payload, args.chunk_size, args.timeout, progress)
            print(f'\nPASS: every echoed byte matched; {time.monotonic()-started:.2f} s elapsed.')
            print('This verifies transport only, not tensor decoding or neural inference.')
    except (serial.SerialException, TimeoutError, ValueError) as error:
        parser.exit(1, f'\nFAIL: {error}\nReset the FPGA before retrying.\n')


if __name__ == '__main__':
    main()
