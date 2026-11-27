#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PIN_FILE="${PIN_FILE:-upduino_spi.pcf}"
mkdir -p build
yosys -l build/synth.log -p '
    read_verilog -I. -sv rtl/spi_byte_link.sv rtl/snn_v2_core.sv rtl/snn_v2_stream.sv rtl/top_upduino_spi.sv
    synth_ice40 -top top_upduino_spi -dsp -json build/top_upduino_spi.json
    check -assert
    stat
'
nextpnr-ice40 --up5k --package sg48 --freq 12 \
    --pcf "$PIN_FILE" --json build/top_upduino_spi.json --asc build/top_upduino_spi.asc \
    2>&1 | tee build/pnr.log
icepack build/top_upduino_spi.asc build/top_upduino_spi.pending.bin
test -s build/top_upduino_spi.pending.bin
mv build/top_upduino_spi.pending.bin build/top_upduino_spi.bin
echo 'Built build/top_upduino_spi.bin'
