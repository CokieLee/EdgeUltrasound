# Cmod S7: stream a tensor from a PC

This is an initial UART transport project for the Digilent Cmod S7-25 Rev. B,
using the already-established Yosys / Xilinx nextpnr / fpga-as toolchain.
It receives bytes through the board's USB-UART bridge, exposes them inside the
FPGA, and echoes them to the PC for exact comparison.

**This is transport only: no CNN, tensor decoding, float arithmetic, learned
weights, quantization, or full-tensor storage is implemented.**

## Why streaming works

One UART RX signal carries successive bits. You do not need a pin for every
tensor element. The same micro-USB connector provides the FTDI programming and
UART functions; no extra jumper wires or USB-to-UART adapter are needed.

The requested tensor shape `[1, 384, 32, 156]` contains 1,916,928 values:

| Wire representation | Bytes per tensor | Ideal send time at 115200 baud, 8N1 |
| --- | ---: | ---: |
| float32 | 7,667,712 | 665.6 s (11.1 min) |
| int16 | 3,833,856 | 332.8 s (5.5 min) |
| int8 | 1,916,928 | 166.4 s (2.8 min) |

Echo/block checking adds overhead. This conservative baud rate is for bring-up,
not a production throughput target. If you use one receive event instead, divide
those sizes and ideal times by 384. This project does not silently select an event
or change the tensor's shape.

The FPGA has a 256-byte echo FIFO and does not hold the complete tensor. A future
CNN consumer must process or buffer arriving data as needed. Streaming avoids
full-input storage but does not eliminate convolution/state buffering needs.

## Included files

| File | Purpose |
| --- | --- |
| `rtl/uart_rx.sv` | Two-stage input synchronizer and 8N1 UART receiver |
| `rtl/uart_tx.sv` | Handshaked 8N1 UART transmitter |
| `rtl/uart_byte_link.sv` | Byte output, echo FIFO, byte counter, sticky errors |
| `rtl/top.sv` | Board reset/LED wrapper; future decoder connection point |
| `cmod_s7_uart.xdc` | Clock, button, UART and LED pins |
| `host/send_tensor.py` | Known-pattern test or .npy tensor transfer with echo checking |
| `host/test_send_tensor.py` | Host-side serialization/error tests |
| `sim/tb_uart_link.sv` | Bit-level UART simulation testbench |
| `build.sh` | Synthesis, routing and assembly with the known local paths |

## 1. Simulate first

In WSL with Icarus Verilog available:

```bash
bash sim/run.sh
```

If needed on Ubuntu:

```bash
sudo apt install iverilog
```

The testbench exercises all byte values, longer transfers with FIFO pointer wrap,
small baud-period differences, invalid stop-bit detection, and reset/restart. A
timeout or mismatched byte fails the simulation. The testbench is not synthesis input.

The design uses a 12 MHz input clock and 115200 baud with an integer divider of
104 clocks per bit (actual FPGA baud approximately 115385, a +0.16% error).
Keep at least four clock cycles per bit if changing parameters; counter width
is 16 bits. Change both FPGA parameters and the PC baud setting together.

## 2. Build in WSL

Run from this extracted project directory:

```bash
bash build.sh
```

The script uses these existing installation paths:

- `$HOME/tools/openxc7/bin/nextpnr-himbaechel`
- `$HOME/tools/openxc7/chipdb/xc7s25.bin`
- `$HOME/fpga-assembler/bazel-bin/fpga/fpga-as`
- `$HOME/prjxray-db/spartan7` (the database that previously worked with fpga-as)

Override `NEXTPNR`, `CHIPDB`, `FPGA_AS`, or `XRAY_DB` if your paths differ.
The script stops on failures and only promotes a nonempty assembled output to
`build/top.bit`. This is a new top module; build in this directory so it does not
replace the previous CNN/blink sources.

To validate synthesis separately before proceeding:

```bash
mkdir -p build
yosys -l build/synth.log -p 'read_verilog -sv rtl/uart_rx.sv rtl/uart_tx.sv rtl/uart_byte_link.sv rtl/top.sv; synth_xilinx -family xc7 -top top; check -assert; write_json build/top.json'
```

## 3. Upload

```bash
openFPGALoader -b cmod_s7 build/top.bit
```

Use the existing usbipd attach procedure if programming in WSL and the FTDI device
is not attached. Check the current BUSID rather than assuming it remains 1-4.

**Press and release button 0 after programming and before a new transfer.** The
button asserts reset asynchronously, with synchronous release inside the wrapper.
Do not reset during a transfer.

| LED | Meaning |
| --- | --- |
| `led[0]`, E2 | Heartbeat, changes state about every 0.70 seconds |
| `led[1]`, K1 | At least one byte received since reset |
| `led[2]`, J1 | Sticky UART framing error or echo FIFO overflow; should stay off |
| `led[3]`, E1 | Received-byte counter bit 8; toggles every 256 bytes |

LEDs alone do not verify data correctness. The host's exact echo comparison does.

## 4. Choose the PC environment and UART port

Install dependencies in the same Python environment used to send the data:

```bash
python -m pip install -r requirements.txt
python host/send_tensor.py --list-ports
```

Use the FTDI UART/VCP port belonging to the board, not a generic JTAG interface.
Examples below use `python`; on your WSL installation that may be `python3`.

### Windows sender (often simplest for UART bring-up)

Program the FPGA using the working WSL flow, then detach the FTDI device in
Windows PowerShell so Windows can access its serial port:

```powershell
usbipd list
usbipd detach --busid 1-4
```

Substitute the current BUSID. Detaching the USB bridge does not intentionally
erase FPGA SRAM configuration, but unplugging power does. List the serial ports
again, then send using the actual COM port, for example `COM5`.
If no UART port appears, check Device Manager for the FTDI VCP driver.

### WSL sender

Leave the device attached to WSL and select its actual `/dev/ttyUSB*` port. Do not
assume the UART is always `/dev/ttyUSB1`. If no device appears, the WSL FTDI serial
driver or device permissions may need setup; native Windows sending avoids that
particular dependency. Only one process should open the UART at a time.

## 5. Run a small, verified link test

On Windows, replacing COM5 with the actual port:

```powershell
python host/send_tensor.py --port COM5
```

Or in WSL, replacing the device path as needed:

```bash
python3 host/send_tensor.py --port /dev/ttyUSB1
```

This sends a 1024-byte known pattern and compares every echo byte. Require a
`PASS` result and an unlit error LED before attempting a tensor. Close serial
terminal programs that could consume the echoed bytes.

## 6. Export and send a small piece of your actual tensor

In the Python session where `x_sample` is already available:

```python
import numpy as np

x = x_sample.detach().cpu().contiguous().numpy()
print(x.shape, x.dtype)
# Transport test only: one event, one channel, first 32 samples.
np.save("input_small.npy", x[:, :1, :1, :32])
# Save the original full tensor too; no event selection or quantization here.
np.save("input_full.npy", x)
```

Then, from the sender environment (file path must be accessible there):

```bash
python host/send_tensor.py --port COM5 --file input_small.npy
```

After that passes, optionally send `input_full.npy`. Check the printed wire time
first: a full float32 tensor at the initial baud rate takes over 11 minutes.
No need to transfer the whole tensor just to validate the initial receiver.

## Wire format and future neural-network connection

- UART: 115200 baud, 8 data bits, no parity, one stop bit; no RTS/CTS flow control.
- Bytes: raw C-order tensor payload, little-endian for multi-byte values.
- For `[batch,event,channel,sample]`, sample varies fastest, then channel, then event.
- `.npy` headers are NOT transmitted; the host loads the file and sends only array data.
- Supported source dtypes: float32, int16, int8, uint8. No automatic numeric conversion.
- Float32 is transported as its four IEEE-754 bytes; the FPGA does not interpret or
  calculate with them. A future fixed-point engine will need calibrated integer inputs.
- The host sends up to 256 bytes, waits for the complete echo, checks it, then continues.
  This paces this receiver only. The echo is not an inference-completion acknowledgement.
- There are no packet headers, shape/dtype fields, sequence numbers, frame-boundary
  markers, or retransmission rules in the FPGA yet. Reset and sender ordering define
  the test session. After an error, reset before retrying.

Inside `top.sv`, `byte_data[7:0]` and `byte_valid` are the connection point for a
future decoder. The consumer must accept every valid byte immediately; it has no
ready/backpressure signal. The FIFO buffers outgoing echoes, not stalled CNN input.
Before adding a slow CNN consumer, add input buffering and an acknowledgement/credit
protocol so the PC cannot overwrite unprocessed data. An int16 decoder will combine
successive low/high bytes and use counters to reconstruct tensor positions.

## Validation and limitations

Host tests:

```bash
python -m unittest discover -s host -p 'test_*.py' -v
```

The six host tests cover partial writes/reads, all byte values, final short blocks,
corruption, missing echoes, invalid chunk sizes, wire endianness, and dtype rejection.
They do not replace RTL simulation or board testing. See `VALIDATION.md` for what
was actually run when this package was created.

## Pin reference

[Digilent Cmod S7-25 master XDC](https://github.com/Digilent/digilent-xdc/blob/master/Cmod-S7-25-Master.xdc)

The UART names are from the FPGA perspective: RX=K15, TX=L12. Your PC sends into
FPGA RX; the FPGA echoes through TX. The 12 MHz clock is M9 and button 0 is D2.
