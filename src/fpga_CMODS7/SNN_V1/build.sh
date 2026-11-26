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
PNR_TIMEOUT="${PNR_TIMEOUT:-12m}"
mkdir -p build
yosys -l build/synth.log -p 'read_verilog -sv rtl/uart_rx.sv rtl/uart_tx.sv rtl/lif_step.sv rtl/snn_engine.sv rtl/top_snn.sv; synth_xilinx -family xc7 -top top_snn; check -assert; stat; write_json build/top_snn.json'
python3 check_bram.py build/top_snn.json
echo "Running placement and routing with a $PNR_TIMEOUT limit."
if timeout --foreground "$PNR_TIMEOUT" "$NEXTPNR" \
    --device xc7s25csga225-1 --chipdb "$CHIPDB" --json build/top_snn.json \
    --freq 12 -o xdc=board.xdc -o fasm=build/top_snn.fasm --log build/pnr.log; then
    "$FPGA_AS" --prjxray_db_path="$XRAY_DB" --part=xc7s25csga225-1 \
        < build/top_snn.fasm > build/top_snn.pending.bit
    test -s build/top_snn.pending.bit
    mv build/top_snn.pending.bit build/top_snn.bit
else
    status=$?
    echo "nextpnr exited with status $status. See build/pnr.log. No new bitstream was assembled." >&2
    exit "$status"
fi
echo 'Built build/top_snn.bit'
