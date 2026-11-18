#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p build
iverilog -g2012 -s tb_uart_link -o build/tb_uart_link \
    rtl/uart_rx.sv rtl/uart_tx.sv rtl/uart_byte_link.sv sim/tb_uart_link.sv
vvp build/tb_uart_link
