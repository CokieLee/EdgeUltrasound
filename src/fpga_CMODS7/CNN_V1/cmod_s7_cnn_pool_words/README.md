# Cmod S7 pooling accumulator diagnostic

This is a separate copy of the sequential-address streaming CNN. In streaming mode the FPGA
reports each channel's signed 48-bit accumulator and its signed 16-bit pooled
output. The CNN engine emits the quotient and three accumulator words through
the existing 16-bit trace port. It holds the accumulator until all four words
are accepted. The UART wrapper only changes its version byte. Pooling address,
accumulation and division are the same as in the sequential-address build.

## Run on the board

Extract this folder beside the current streaming project. Copy the existing
`demo` directory containing the exact weights, configuration and input that gave
the reference score of -50 and the hardware score of -73.

From this folder in WSL:

```bash
bash sim/run.sh &&
PNR_TIMEOUT=12m bash build.sh stream &&
openFPGALoader -b cmod_s7 build/top_stream.bit
```

The optional UART simulation (`bash sim/run_uart.sh`) may take several minutes. It verifies the exact emitted
bytes, including all 32 pooling sums. The placement and routing limit gives the
placer more time because the previous build took over five minutes.

Detach the board from WSL, keep it powered, and press and release button 0.
From this folder in Windows PowerShell:

```powershell
python host/run_stream.py --port COM7 --require-pool-sums
```

Use the current COM port. If dependencies are missing, install them with
`python -m pip install -r requirements.txt`. Use the receiver from this folder.
The FPGA identity is `B5 02`. The updated receiver can also parse old captures, and
`--require-pool-sums` checks the new identity before uploading the weights.

## Read the results

Open `report.json` in the new timestamped folder under `stream_results`. The
`pool_diagnostic` section includes per-channel expected and observed sums,
observed pooled values, and the quotient Python computes from the observed sum.
`pool_sums.npy` contains the same 32 raw sums as signed int64 values.

| Comparison | What a mismatch implicates |
| --- | --- |
| `sum_vs_emitted_conv3` | Pooling RAM reads or accumulation |
| `quotient_vs_fpga_sum_divided_by_80` | Division or its output path |

For the original demo, channel 0's expected sum is 7,581 and expected quotient is
94. The host saves completed reports even when numerical checks fail. If the
trace agrees with Python throughout, compare with continuous mode using
`python host/run_stream.py --port COM7 --continuous` after resetting. The new
mapping and the extra time spent transmitting may change whether the original
hardware discrepancy reproduces.

## Wire format

After the single `C4` tag, there are 32 records in channel order. Each record
contains a little-endian int16 quotient followed by a little-endian signed
48-bit sum. The rest of the streaming response and the continuous `I` exchange
use the existing layout. The stream still has no per-layer CRC.

The Python tests and decode of a generated reference capture passed during
preparation. HDL simulation, placement and board verification require your FPGA
toolchain.
