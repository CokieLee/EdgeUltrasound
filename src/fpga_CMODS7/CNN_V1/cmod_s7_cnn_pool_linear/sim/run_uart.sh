#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python3 sim/make_vectors.py
iverilog -g2012 -s tb_uart_stream -o build/tb_uart_stream rtl/*.sv sim/tb_uart_stream.sv
vvp build/tb_uart_stream
python3 host/run_stream.py --capture build/uart_capture.bin \
    --weights build/vectors/demo_weights.npz --config build/vectors/demo_config.json \
    --input build/vectors/demo_input.npy
python3 - <<'PY'
from pathlib import Path
actual=Path('build/uart_capture.bin').read_bytes()
expected=Path('build/vectors/reference_capture.bin').read_bytes()
if actual!=expected:
    raise SystemExit('Actual simulated UART bytes differ from Python reference')
print('PASS: all actual simulated UART bytes match the reference.')
PY
