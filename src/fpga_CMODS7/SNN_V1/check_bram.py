#!/usr/bin/env python3
"""Verify activation and parameter banks use BRAM with data-only lanes."""
from collections import Counter
import json
from pathlib import Path
import sys


def verify(netlist):
    cells = netlist['modules']['snn_engine']['cells']
    consumers = {bit for cell in cells.values()
                 for port, bits in cell['connections'].items()
                 if cell['port_directions'][port] in ('input', 'inout')
                 for bit in bits if isinstance(bit, int)}
    counts = Counter()
    for bank in ("weights_lo", "weights_hi", "original_lo", "original_hi",
                 "stem_lo", "stem_hi", "spikes1", "spikes2"):
        # Accept Yosys generated bank indices without depending on their count.
        for name, cell in cells.items():
            if not name.startswith(bank + '.'):
                continue
            if cell['type'] not in ('RAMB18E1', 'RAMB36E1'):
                raise ValueError(f'{name} maps to {cell["type"]}, expected block RAM')
            counts[bank] += 1
            pins = cell['connections']
            for port in ('DIPADIP', 'DIPBDIP'):
                if any(isinstance(bit, int) for bit in pins[port]):
                    raise ValueError(f'{name}.{port} has a live write-data connection')
            for port in ('DOPADOP', 'DOPBDOP'):
                if consumers.intersection(pins[port]):
                    raise ValueError(f'{name}.{port} has a live read-data consumer')
        if not counts[bank]:
            raise ValueError(f'No synthesized BRAM cells found for {bank}')
    return counts


if __name__ == '__main__':
    try:
        counts = verify(json.loads(Path(sys.argv[1]).read_text()))
    except (ValueError, KeyError) as exc:
        sys.exit(f'BRAM layout check failed: {exc}')
    print(f'BRAM layout check passed: {dict(counts)}. No bank uses parity data pins.')
