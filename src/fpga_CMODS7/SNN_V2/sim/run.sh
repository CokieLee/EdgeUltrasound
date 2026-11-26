#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python3 sim/vectors.py
iverilog -g2012 -I. -s tb_uart_v2 -o build/tb_uart_v2 rtl/*.sv sim/tb_uart_v2.sv
vvp build/tb_uart_v2
