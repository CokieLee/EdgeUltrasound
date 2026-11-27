#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python3 sim/vectors.py
iverilog -g2012 -I. -s tb_spi -o build/tb_spi rtl/*.sv sim/tb_spi.sv
vvp build/tb_spi
iverilog -g2012 -I. -s tb_power -o build/tb_power rtl/*.sv sim/tb_power.sv
vvp build/tb_power
