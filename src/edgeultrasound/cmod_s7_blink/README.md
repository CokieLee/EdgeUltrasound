# Cmod S7 blink: RTL-to-FPGA pipeline
**Target FPGA:** `xc7s25csga225-1` (Spartan-7).

## Status

The session verified synthesis, placement and routing at a 12 MHz target, and generation of a 1,241,907-byte bitstream from the newly routed FASM. Bitstream upload was previously confirmed as a separate operation. The final LED behavior of this newly built design still needs to be recorded.

Matching file sizes do not prove that two bitstreams have identical contents or that the design functions correctly.

## Suggested repository layout

Put this README and the example sources in `src/examples/cmod_s7_blink/`:

| File | Purpose |
| --- | --- |
| `README.md` | This guide |
| `top.v` | Verilog source below |
| `cmod_s7_nextpnr.xdc` | nextpnr-compatible pin constraints |
| `top.sv` | Optional SystemVerilog alternative; select one top source |

Keep generated netlists, logs, FASM and bitstreams out of Git unless the team intentionally archives build results. Do not commit the installed tools or generated chip database with the example.

## Pipeline

| Stage | Tool | Input | Output |
| --- | --- | --- | --- |
| Synthesis | Yosys | `top.v` | `top.json` |
| Placement and routing | Xilinx-capable nextpnr | `top.json`, XDC, chip database | `top.fasm`, `pnr.log` |
| Bitstream assembly | fpga-as | `top.fasm`, Project X-Ray database | `top.pending.bit` |
| Programming | openFPGALoader | `top.pending.bit` | FPGA configuration in SRAM |

Synthesis maps the RTL to FPGA resources. Placement and routing selects physical sites and wires while applying pin and timing constraints. FASM describes configuration features; fpga-as converts those features into the binary bitstream.

## Example source: top.v 

```verilog
module top (
    input  wire clk,
    input  wire rst,
    output reg  led
);
    reg [23:0] count;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            count <= 24'd0;
            led   <= 1'b0;
        end else begin
            count <= count + 1'b1;
            led   <= count[23];
        end
    end
endmodule
```

``` systemverilog
module top (
    input  logic clk,
    input  logic rst,
    output logic led
);
    logic [23:0] count;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            count <= 24'd0;
            led   <= 1'b0;
        end else begin
            count <= count + 24'd1;
            led   <= count[23];
        end
    end
endmodule
```

## Pin constraints: cmod_s7_nextpnr.xdc

```tcl
set_property PACKAGE_PIN M9 [get_ports {clk}]
set_property IOSTANDARD LVCMOS33 [get_ports {clk}]
set_property PACKAGE_PIN D2 [get_ports {rst}]
set_property IOSTANDARD LVCMOS33 [get_ports {rst}]
set_property PACKAGE_PIN E2 [get_ports {led}]
set_property IOSTANDARD LVCMOS33 [get_ports {led}]
```

| Top-level port | Board signal | Package pin |
| --- | --- | --- |
| `clk` | 12 MHz oscillator | M9 |
| `rst` | Push button 0 | D2 |
| `led` | First discrete LED | E2 |

Port names connect the constraints to the selected top module; the source and XDC filenames do not need to match.

## Prerequisites and paths

The commands assume the installation layout used during bring-up:

| Path | Purpose |
| --- | --- |
| `$HOME/tools/openxc7/export.sh` | Toolchain environment |
| `$HOME/tools/openxc7/bin/nextpnr-himbaechel` | Build with the Xilinx uarch enabled |
| `$HOME/tools/openxc7/chipdb/xc7s25.bin` | Previously generated nextpnr chip database |
| `$HOME/fpga-assembler/bazel-bin/fpga/fpga-as` | Previously built assembler |
| `$HOME/prjxray-db/spartan7` | Database that worked with this assembler |

Yosys and openFPGALoader must also be available. Adapt paths for each teammate's installation.

The original OSS CAD Suite nextpnr build only included GateMate and Gowin, so a separate Xilinx-capable build was installed. Verify the intended executable:

```bash
"$HOME/tools/openxc7/bin/nextpnr-himbaechel" --list-uarch
```
Its output must include `xilinx`. A generic nextpnr installation is not sufficient.

## 1. Set up the terminal

Run in WSL. Change into the directory containing the example sources.

For the original working directory:

```bash
cd "$HOME/Neuromorphic"
source "$HOME/tools/openxc7/export.sh"
```

For a repository checkout, use the corresponding `src/examples/cmod_s7_blink` directory instead.

## 2. Synthesize

```bash
yosys -l synth.log -p \
  'read_verilog top.v; synth_xilinx -family xc7 -top top; check; write_json top.json'
```

alternatively for the system verilog script you can do:
```bash
yosys -l synth.log -p \
  'read_verilog -sv top.sv; synth_xilinx -family xc7 -top top; check; write_json top.json'
```

Validate that Yosys finishes without errors, writes `top.json`, and reports zero problems in its CHECK pass.

```bash
tail -n 40 synth.log
```

The expected resources include 25 flip-flops (24 counter bits plus the LED register), carry logic and I/O/clock buffers. Exact statistics can vary with tool versions.

## 3. Place and route; write FASM

```bash
"$HOME/tools/openxc7/bin/nextpnr-himbaechel" \
  --device xc7s25csga225-1 \
  --chipdb "$HOME/tools/openxc7/chipdb/xc7s25.bin" \
  --json top.json \
  -o xdc=cmod_s7_nextpnr.xdc \
  --freq 12 \
  -o fasm=top.fasm \
  --log pnr.log
```

Validate:

```bash
ls -lh top.fasm
head -n 10 top.fasm
tail -n 40 pnr.log
grep -in -C 2 'warning' pnr.log
```

The observed run finished normally with zero errors and PASS at 12 MHz. It warned that the internal clock had no explicit timing constraint and used the default 12 MHz supplied by `--freq 12`.

The reported maximum frequency (402.74 MHz in that run) is a tool estimate, not the board oscillator frequency or a hardware-verified operating limit. That setup-timing result does not establish complete hold-time or asynchronous reset timing closure.

## 4. Assemble the bitstream

Use the original database that worked with fpga-as:

```bash
if "$HOME/fpga-assembler/bazel-bin/fpga/fpga-as" \
  --prjxray_db_path="$HOME/prjxray-db/spartan7" \
  --part=xc7s25csga225-1 \
  < top.fasm > top.pending.bit
then
  if test -s top.pending.bit; then
    wc -c < top.pending.bit
  else
    echo "Assembly produced an empty file; do not upload."
  fi
else
  echo "Assembly failed; do not upload."
fi
```

The observed output size was **1,241,907 bytes**. A nonempty file is a basic check, not proof of correct hardware behavior.

The database installed under `$HOME/tools/openxc7/share/nextpnr/prjxray-db/spartan7` failed with this assembler because its part JSON lacked expected IDCODE and frame-layout metadata. The original `$HOME/prjxray-db/spartan7` assembled this design without reporting an error.

This records the combination used during bring-up, not a guarantee that arbitrary database revisions can be mixed. Pin and record both database and tool revisions for reproducible team builds.

Shell redirection creates or truncates the output before the program runs. A failed assembly can therefore leave a zero-byte or partial file. Upload only after successful assembly and a nonempty-output check. If using `echo $?`, run it immediately after the command being checked; any intervening command replaces that status.

## 5. Connect the USB device to WSL, if needed

In **Windows PowerShell**:

```powershell
usbipd list
```

During bring-up, the board appeared as `0403:6010 USB Serial Converter A/B` at BUSID `1-4`. The BUSID can change; use the current board's ID, especially if several FTDI devices are connected.

If already Shared:

```powershell
usbipd attach --wsl --busid 1-4
```

If Not shared, first run this in **PowerShell as Administrator**:

```powershell
usbipd bind --busid 1-4
```

Then run the attach command. If already Attached, proceed to programming.

## 6. Upload and check hardware behavior

In **WSL**, from the example directory:

```bash
openFPGALoader -b cmod_s7 top.pending.bit
```

Check the loader's success report and observe the LED. Expected behavior is a state change about every 0.7 seconds; holding button 0 should drive the LED off.

This programs volatile FPGA configuration memory. The configuration is lost when power is removed; this command does not persist the design to flash.

Record the final loader output and observed LED/reset behavior here when hardware validation is complete.

## Using SystemVerilog

SystemVerilog is suitable for this example. Yosys's built-in frontend supports a subset, including `logic`, `always_ff`, and `always_comb`, when enabled with `read_verilog -sv`. It is not a full SystemVerilog frontend.

Save the equivalent source as `top.sv`:

```systemverilog
module top (
    input  logic clk,
    input  logic rst,
    output logic led
);
    logic [23:0] count;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            count <= 24'd0;
            led   <= 1'b0;
        end else begin
            count <= count + 1'b1;
            led   <= count[23];
        end
    end
endmodule
```

Replace the synthesis command with:

```bash
yosys -l synth.log -p \
  'read_verilog -sv top.sv; synth_xilinx -family xc7 -top top; check; write_json top.json'
```

Select either `top.v` or `top.sv`; do not load both definitions of `module top`. The constraints and downstream commands stay the same because the top-level ports are unchanged.

This SystemVerilog version is a proposed equivalent; the session validated the Verilog version. Re-run synthesis and the downstream flow after switching.

For multiple source files, pass each required RTL file to `read_verilog -sv`, reading package definitions before files that import them. Keep simulation testbenches separate from synthesis inputs. Check more advanced SystemVerilog constructs against the installed frontend's support before adopting them across the project.

## Rebuilding and reproducibility

- RTL changes: repeat synthesis, placement/routing, assembly and upload.
- Pin or timing constraint changes: repeat placement/routing, assembly and upload.
- Tool installation and chip database generation: normally one-time setup for the selected tool/device combination; revisit when tool/database revisions change.

For the team handoff, record:

```bash
yosys -V
"$HOME/tools/openxc7/bin/nextpnr-himbaechel" --version
openFPGALoader --version
git -C "$HOME/fpga-assembler" rev-parse HEAD
git -C "$HOME/prjxray-db" rev-parse HEAD
git -C "$HOME/tools/openxc7-build/nextpnr" rev-parse HEAD
git -C "$HOME/tools/openxc7-build/prjxray-db" rev-parse HEAD
```

Keep these outputs with build notes. Newer or different revisions may change behavior and database compatibility.

## References

- [fpga-assembler](https://github.com/lromor/fpga-assembler)
- [openXC7 toolchain installer](https://github.com/openXC7/toolchain-installer)
- [openXC7 Project X-Ray database](https://github.com/openXC7/prjxray-db)
- [Digilent Cmod S7-25 master XDC](https://github.com/Digilent/digilent-xdc/blob/master/Cmod-S7-25-Master.xdc)
- [Yosys SystemVerilog support](https://yosyshq.readthedocs.io/projects/yosys/en/latest/using_yosys/verilog.html)
- [WSL USB device connection](https://learn.microsoft.com/en-us/windows/wsl/connect-usb)

