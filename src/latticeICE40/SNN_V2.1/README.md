# SNN V2.1 on UPduino v3.1 over the onboard FTDI

Use one micro-USB cable for both flash programming and SNN inference. The 12 MHz SNN core and the checkpoint files are byte-identical to the Cmod S7 version. The command bytes, 128-byte tokens, per-token credit, optional traces, input CRC, and signed Q20 result are the same. The transport uses SPI through the onboard FT232H in MPSSE mode.

## Board and clock

Use the factory SPI-flash wiring, with R12/R13 in their standard positions. The bitstream expects the onboard 12 MHz oscillator routed to GPIO 20. Reuse your existing OSC/R16 connection or jumper `12M` to `20` with power off. The PCF numbers are FPGA package pins.

| Onboard signal | FPGA pin | Role during inference |
|---|---:|---|
| FTDI SCK | 15 | SPI clock input |
| FTDI MOSI / flash DI / FPGA SO | 14 | FPGA data input |
| FTDI MISO / flash DO / FPGA SI | 17 | FPGA data output |
| Flash chip-select | 16 | FPGA input, normally high |

The FPGA releases MISO immediately when flash select goes low. The host waits for CDONE and inactive flash select before using the bus. To reset the application protocol, it briefly pulls flash select low while holding SCK low, with no SPI clocks. It then releases flash select to the board's pull-up and uses the bus for inference while the flash stays deselected. End the inference process before running `iceprog` again.

## Build and flash in WSL

Attach the UPduino to WSL as for your working `iceprog` setup. Use the same OSS CAD Suite environment.

```bash
python3 -m pip install -r requirements.txt
python3 sim/test_host.py
python3 sim/test_ftdi.py
bash sim/run.sh
bash build.sh
iceprog build/top_upduino_spi.bin
```

The build uses DSP inference and targets 12 MHz on the UP5K SG48. The SPI simulation uses a 250 kHz clock and checks all 2,220 expected response bytes over three frames. It also checks that the FPGA releases MISO while flash is selected.

## Run over the same USB cable

Keep the UPduino attached to WSL. The host opens the FTDI USB interface through PyFtdi/libusb. `COM16` belongs to Windows' serial driver, and this SPI transport uses the FTDI USB device directly. Close any other process accessing it. If PyFtdi cannot see the device, verify USB attachment with `lsusb -d 0403:6014`, and use the same USB permissions that let you run `iceprog`.

```bash
python3 host/run_snn_v2.py --ftdi --dummy --trace
tokens='/mnt/c/Users/ethan/Documents/Duke/Classes/Neuromorphic/Stanford_Murine_Processed_RFData_SNNV2/Tokens/float32'
python3 host/run_snn_v2.py --ftdi --normalized "$tokens/rat01_frame00.npy" --trace
python3 host/run_snn_v2.py --ftdi --normalized "$tokens" --output upduino_real.json
```

The dummy should return `-1055882` Q20. The Cmod S7 returned `-808280` Q20 for `rat01_frame00.npy`. Traced runs check both layers' spike masks and voltages. Every device run checks the input CRC and Q20 score. If there are multiple FT232H devices, provide a PyFtdi URL with the serial number as the value of `--ftdi`.

The SPI wire format is mode 0, MSB-first, with two-byte slots. Each direction carries a flag byte (0 for no data and 1 for a valid byte), then a data byte. The FTDI clocks these slots at 250 kHz. The FPGA snapshots pending responses at the first clock of each slot and acknowledges a byte only once the slot is complete. The host sends one token at a time and polls for its response before sending the next one.

The Python tests and reference vector generation passed. HDL simulation, nextpnr timing, and physical MPSSE communication still require verification. This build supports streaming verification. Its transport waits between tokens, so a sustained core-power experiment needs its own on-device repeat loop.
