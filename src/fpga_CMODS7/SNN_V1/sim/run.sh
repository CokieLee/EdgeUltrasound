#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python3 sim/make_vectors.py
iverilog -g2012 -s tb_lif -o build/tb_lif rtl/lif_step.sv sim/tb_lif.sv
vvp build/tb_lif
case "${1:-engine}" in
    engine)
        iverilog -g2012 -s tb_snn_engine -o build/tb_snn_engine rtl/*.sv sim/tb_snn_engine.sv
        vvp build/tb_snn_engine
        ;;
    uart)
        iverilog -g2012 -s tb_uart_snn -o build/tb_uart_snn rtl/*.sv sim/tb_uart_snn.sv
        vvp build/tb_uart_snn
        cmp build/uart_capture.bin build/vectors/expected_uart.bin
        echo 'PASS UART capture matches the Python reference byte for byte'
        ;;
    *) echo 'Usage: bash sim/run.sh [engine|uart]' >&2; exit 2 ;;
esac
