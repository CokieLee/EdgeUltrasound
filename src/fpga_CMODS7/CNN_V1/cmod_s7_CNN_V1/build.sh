#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ -f "$HOME/tools/openxc7/export.sh" ]]; then
    source "$HOME/tools/openxc7/export.sh"
fi
mkdir -p build
yosys -l build/synth.log -p 'read_verilog -sv rtl/uart_rx.sv rtl/uart_tx.sv rtl/cnn_engine.sv rtl/top.sv; synth_xilinx -family xc7 -top top; check -assert; stat; write_json build/top.json'
"$HOME/tools/openxc7/bin/nextpnr-himbaechel" --device xc7s25csga225-1 \
    --chipdb "$HOME/tools/openxc7/chipdb/xc7s25.bin" --json build/top.json \
    -o xdc=cmod_s7_cnn.xdc --freq 12 -o fasm=build/top.fasm --log build/pnr.log
"$HOME/fpga-assembler/bazel-bin/fpga/fpga-as" \
    --prjxray_db_path="$HOME/prjxray-db/spartan7" --part=xc7s25csga225-1 \
    < build/top.fasm > build/top.pending.bit
test -s build/top.pending.bit
mv build/top.pending.bit build/top.bit
echo 'Built build/top.bit. Upload separately with openFPGALoader.'
