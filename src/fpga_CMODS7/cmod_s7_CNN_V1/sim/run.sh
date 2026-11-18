#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python3 sim/make_vectors.py
iverilog -g2012 -s tb_cnn_engine -o build/tb_cnn_engine rtl/cnn_engine.sv sim/tb_cnn_engine.sv
vvp build/tb_cnn_engine
# Elaborate the board wrapper too (the long test above exercises the CNN core).
iverilog -g2012 -s top -o build/top_elaboration rtl/*.sv
