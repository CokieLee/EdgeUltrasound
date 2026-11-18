#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ -f "$HOME/tools/openxc7/export.sh" ]]; then
    source "$HOME/tools/openxc7/export.sh"
fi
NEXTPNR="${NEXTPNR:-$HOME/tools/openxc7/bin/nextpnr-himbaechel}"
CHIPDB="${CHIPDB:-$HOME/tools/openxc7/chipdb/xc7s25.bin}"
FPGA_AS="${FPGA_AS:-$HOME/fpga-assembler/bazel-bin/fpga/fpga-as}"
XRAY_DB="${XRAY_DB:-$HOME/prjxray-db/spartan7}"
mkdir -p build
yosys -l build/synth.log -p 'read_verilog -sv rtl/uart_rx.sv rtl/uart_tx.sv rtl/uart_byte_link.sv rtl/top.sv; synth_xilinx -family xc7 -top top; check -assert; write_json build/top.json'
"$NEXTPNR" --device xc7s25csga225-1 --chipdb "$CHIPDB" \
    --json build/top.json -o xdc=cmod_s7_uart.xdc --freq 12 \
    -o fasm=build/top.fasm --log build/pnr.log
"$FPGA_AS" --prjxray_db_path="$XRAY_DB" --part=xc7s25csga225-1 \
    < build/top.fasm > build/top.pending.bit
test -s build/top.pending.bit
mv build/top.pending.bit build/top.bit
ls -lh build/top.bit
echo 'Build completed. Upload separately with:'
echo 'openFPGALoader -b cmod_s7 build/top.bit'
