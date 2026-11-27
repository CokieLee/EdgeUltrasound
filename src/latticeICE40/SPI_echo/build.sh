#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p build
yosys -l build/probe_synth.log -p 'read_verilog -sv probe.sv; synth_ice40 -top probe -json build/probe.json; check -assert'
nextpnr-ice40 --up5k --package sg48 --freq 12 --pcf pins.pcf \
    --json build/probe.json --asc build/probe.asc 2>&1 | tee build/probe_pnr.log
icepack build/probe.asc build/probe.bin
