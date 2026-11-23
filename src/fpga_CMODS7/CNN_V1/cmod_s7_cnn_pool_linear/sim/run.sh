#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python3 sim/make_vectors.py
iverilog -g2012 -s tb_cnn_engine_stream -o build/tb_cnn_engine_stream \
    baseline/rtl/cnn_engine.sv rtl/cnn_engine_stream.sv sim/tb_cnn_engine_stream.sv
vvp build/tb_cnn_engine_stream
iverilog -g2012 -s top_stream -o build/top_elaboration rtl/*.sv
