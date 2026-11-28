# UPduino SNN V2 core-rail power benchmark

This separate build uses the same 12 MHz clock, onboard FTDI SPI transport and Q12/Q20 SNN arithmetic as the verified UPduino streaming build. It expands the token buffer to 2 KiB of frame storage. The FPGA can replay the frame without any SPI clocks during measurement. Normal streaming and trace modes are still available in this build.

Use the existing factory flash wiring, the working OSC connection to GPIO 20, and the FT232H attached to WSL. This package uses the same PCF, PyFtdi access and flash flow as the SNN V2 SPI project.

## Build and check

From this directory in WSL with the UPduino tools on PATH:

```bash
python3 -m pip install -r requirements.txt
python3 sim/test_host.py
python3 sim/test_ftdi.py
bash sim/run.sh
bash build.sh
iceprog build/top_upduino_power.bin
python3 host/run_snn_v2.py --ftdi --dummy --trace
```

The traced dummy should return -1055882 Q20 with all trace and CRC checks passing. `sim/run.sh` runs the SPI regression and a reduced-duration active/idle controller test. FPGA synthesis, placement and 12 MHz timing are unverified here.

## Measure the same frame in both modes

```bash
tokens='/mnt/c/Users/ethan/Documents/Duke/Classes/Neuromorphic/Stanford_Murine_Processed_RFData_SNNV2/Tokens/float32'

python3 host/run_snn_v2.py --ftdi --normalized "$tokens/rat01_frame00.npy" \
    --benchmark idle --output upduino_idle.json
python3 host/run_snn_v2.py --ftdi --normalized "$tokens/rat01_frame00.npy" \
    --benchmark active --output upduino_active.json
```

Each command loads one frame and waits two seconds before the measurement window. The host starts no new SPI transfers for the next 30 seconds. Use the meter's mean reading from the middle of the window. Idle uses the same configured, clocked circuit and loaded frame and counts zero inferences. Active repeatedly processes the frame from BRAM, resetting the membrane state for each inference. It runs for at least 30 seconds and finishes the last inference. The FPGA reports the exact elapsed cycles, count, CRC, final score and whether any repeat score differed from the first. Active should return -808280 Q20 on this real frame. The power and timing problem are independent of USB transfer throughput.

Take several alternating idle and active measurements at the same range and rail voltage if the difference is small. Report power at the **FPGA core rail**. To calculate energy from the JSON output and your meter readings:

- Core power in mW = measured core voltage in V × active current in mA.
- Core energy per inference in µJ = 1000 × core power in mW ÷ measured inferences per second.
- Energy above idle in µJ = 1000 × core voltage × (active mA − idle mA) ÷ measured inferences per second.

If you know the readings from a prior run, the host will also calculate these values with `--voltage 1.2 --current-ma ACTIVE_MA --idle-current-ma IDLE_MA`. Omit the idle-current argument for an idle measurement. If the active–idle difference is comparable to measurement variation, report the difference as unresolved. The ESP32's 3.3 V supply-branch measurement covers a different electrical boundary.

The fixed-length clock counters use 12 MHz. The JSON time and throughput calculations assume the measured board oscillator's nominal 12 MHz rate. No board-side ADC or meter control is included. Existing SPI input CRC and reference score checks still apply.
