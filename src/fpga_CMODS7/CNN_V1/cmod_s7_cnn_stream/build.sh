#!/usr/bin/env bash
# Default: minimal stream build. Optional: bash build.sh baseline
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ -f "$HOME/tools/openxc7/export.sh" ]]; then
    source "$HOME/tools/openxc7/export.sh"
fi
NEXTPNR="${NEXTPNR:-$HOME/tools/openxc7/bin/nextpnr-himbaechel}"
CHIPDB="${CHIPDB:-$HOME/tools/openxc7/chipdb/xc7s25.bin}"
FPGA_AS="${FPGA_AS:-$HOME/fpga-assembler/bazel-bin/fpga/fpga-as}"
XRAY_DB="${XRAY_DB:-$HOME/prjxray-db/spartan7}"
PNR_TIMEOUT="${PNR_TIMEOUT:-10m}"
STREAM_BUILD_VARIANT="${1:-stream}"
case "$STREAM_BUILD_VARIANT" in
    stream)
        STREAM_BUILD_TOP=top_stream
        STREAM_BUILD_SOURCES='rtl/cnn_engine_stream.sv rtl/top_stream.sv'
        ;;
    baseline)
        STREAM_BUILD_TOP=top
        STREAM_BUILD_SOURCES='baseline/rtl/cnn_engine.sv baseline/rtl/top.sv'
        ;;
    *) echo 'Usage: bash build.sh [stream|baseline]' >&2; exit 2 ;;
esac
command -v timeout >/dev/null || { echo 'GNU timeout is required.' >&2; exit 1; }
mkdir -p build
STREAM_BUILD_PREFIX="build/top_${STREAM_BUILD_VARIANT}"
yosys -l "build/${STREAM_BUILD_VARIANT}_synth.log" -p "read_verilog -sv rtl/uart_rx.sv rtl/uart_tx.sv $STREAM_BUILD_SOURCES; synth_xilinx -family xc7 -top $STREAM_BUILD_TOP; check -assert; stat; write_json $STREAM_BUILD_PREFIX.json"
echo "Running $STREAM_BUILD_VARIANT placement/routing; wall-clock limit $PNR_TIMEOUT."
if timeout --foreground "$PNR_TIMEOUT" "$NEXTPNR" \
    --device xc7s25csga225-1 --chipdb "$CHIPDB" \
    --json "$STREAM_BUILD_PREFIX.json" -o xdc=cmod_s7_cnn_stream.xdc --freq 12 \
    -o "fasm=$STREAM_BUILD_PREFIX.fasm" --log "build/${STREAM_BUILD_VARIANT}_pnr.log"; then
    :
else
    STREAM_BUILD_STATUS=$?
    if [[ "$STREAM_BUILD_STATUS" -eq 124 ]]; then
        echo "Stopped by the $PNR_TIMEOUT wall-clock limit; this is not a nextpnr legality verdict." >&2
        echo 'The synthesized JSON is retained. Increase PNR_TIMEOUT only if justified by progress.' >&2
    else
        echo "nextpnr exited with status $STREAM_BUILD_STATUS. Assembly skipped." >&2
    fi
    exit "$STREAM_BUILD_STATUS"
fi
"$FPGA_AS" --prjxray_db_path="$XRAY_DB" --part=xc7s25csga225-1 \
    < "$STREAM_BUILD_PREFIX.fasm" > "$STREAM_BUILD_PREFIX.pending.bit"
test -s "$STREAM_BUILD_PREFIX.pending.bit"
mv "$STREAM_BUILD_PREFIX.pending.bit" "$STREAM_BUILD_PREFIX.bit"
python3 - "$STREAM_BUILD_VARIANT" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
variant = sys.argv[1]
rtl = sorted(Path('rtl').glob('*.sv')) if variant == 'stream' else (
    sorted(Path('baseline/rtl').glob('*.sv')) + [Path('rtl/uart_rx.sv'), Path('rtl/uart_tx.sv')])
paths = rtl + [Path('build.sh'), Path('cmod_s7_cnn_stream.xdc'), Path(f'build/top_{variant}.bit')]
result = {'variant': variant, 'sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
Path(f'build/{variant}_build_manifest.json').write_text(json.dumps(result, indent=2) + '\n')
PY
echo "Built $STREAM_BUILD_PREFIX.bit. Program separately with:"
echo "openFPGALoader -b cmod_s7 $STREAM_BUILD_PREFIX.bit"
