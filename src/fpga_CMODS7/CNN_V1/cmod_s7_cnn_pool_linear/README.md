# Cmod S7 CNN with sequential pooling addresses

This variant starts from the working CNNSTREAM1 project. Pooling walks
`buffer_b` from address 0 to 2559 using the existing address register.
It still consumes 80 values per channel, divides the sum by 80, and
uses the existing UART commands and response format.

Extract beside the original project. Copy your existing `demo` folder with
the exact files that returned -73 into this folder.

From this folder in WSL:

```bash
bash sim/run.sh &&
PNR_TIMEOUT=12m bash build.sh stream &&
openFPGALoader -b cmod_s7 build/top_stream.bit
```

The core simulation checks every pooling address and sampled value in
both modes. The full UART test is also available with `bash sim/run_uart.sh`.
Placement and routing need to be verified on your machine.

After programming, detach USB from WSL and reset button 0. From this
folder in Windows PowerShell, run:

```powershell
python host/run_stream.py --port COM7 --output stream_results/linear_stream
```

Reset button 0 again before running continuous mode:

```powershell
python host/run_stream.py --port COM7 --continuous --output stream_results/linear_continuous
```

The expected demo score is -50 in both modes. The streaming report also
checks all 32 pooled values. The protocol string is CNNSTREAM1 because
this variant uses the same wire format. Use the bitstream from this
folder to distinguish it from the earlier CNNSTREAM1 build.

The Python tests and a software model of the read sequence passed during
preparation. The HDL testbench, place and route, and hardware measurements
require the FPGA toolchain and board.
