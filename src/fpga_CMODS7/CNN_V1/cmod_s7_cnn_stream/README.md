# Minimal produced-word CNN streaming

This separate project starts from the original CNN core and UART wrapper. Instead
of adding a general RAM-readback interface, it captures each 16-bit output already
being written by the engine and sends that one word through the existing UART.
The engine waits until the word has been accepted before advancing.

**Source package; no prebuilt bitstream.** Python tests and offline receiver
checks passed during preparation. The new HDL simulations, synthesis, placement,
routing, fit and hardware operation must be checked on your machine. A simpler
design is an experiment, not a guarantee that the placer will succeed.

## What is different from the previous debug build?

| Original CNN | Previous full trace | This minimal stream |
| --- | --- | --- |
| Original computation RAM ports | Additional address selection for RAM readback | Original RAM read/write block retained byte-for-byte |
| Score and input replay | Readback of configuration, parameters, RAM snapshots, score | Produced layer words, followed by original score/input response |
| Small UART controller | Generic frame controller, CRC32 and source selection | Original controller plus word handshake and layer tags |

The UART already serializes one word at a time; neither debug design connects
every neuron to an individual UART wire. The full trace nevertheless changes
RAM-address selection, data selection, control and physical placement. Those
changes can expose tool limitations without exceeding total resource capacity.
We have not established which change caused the placement trouble.

The stream engine adds a 16-bit held word, its stage/valid controls, and states
that wait for acceptance. The UART wrapper sends a single stage tag and then
low/high bytes using its existing transmitter. There is no extra feature-map
storage, new RAM read port, RAM read-address mux, or CRC32 datapath.

**Words are sent as they are produced, rather than waiting for layer completion.**
Python groups them into complete layer tensors. This deliberately avoids a later
scan/readout of RAM. UART backpressure is supported by holding the engine at each
produced word; it substantially lengthens inference.

## What does this measure?

It measures the quantized values offered to the layer-output RAM write ports,
and the values offered to the pooled-vector registers. These are **produced
values, not an independent verification of stored RAM contents**.

For example, correct emitted conv1 values followed by incorrect conv2 values can
implicate the intervening RAM writes/reads, conv2 weights, addressing or arithmetic.
It does not automatically prove that conv2's multiplier is wrong.

This minimal protocol retains the original length acknowledgment for parameters.
It does not read back weights/configuration and does not independently checksum
the intermediate transfer. Repeat any numerical mismatch to assess repeatability;
a corrupted output byte could otherwise resemble a computational error. The
original exact input replay check remains in the final response.

## Setup

Extract `cmod_s7_cnn_stream` beside your existing projects. Copy your original
`demo` folder into it, keeping the same weights/config/input that produced -73.
Do not replace the original or the full-trace project.

The `baseline/rtl` directory contains unchanged copies of the original core and
top from this workspace. UART modules and pin constraints are also unchanged
copies. Baseline and stream builds use the same tool settings and RAM arrays.
If you edited your original RTL locally, check that these baseline files match
the version you previously built before interpreting the comparison.

## Validate (WSL, inside this new project)

```bash
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s host -p 'test_*.py' -v
bash sim/run.sh
bash sim/run_uart.sh
```

The core test checks every produced word against NumPy, holds backpressure to
check word stability, checks preserved input and score, then executes continuous
mode alongside the original core and compares its cycle sequence and result.

The UART test drives the actual wrapper RX pin and receives its TX pin using an
accelerated UART divider. It saves the actual transmitted bytes, passes them
through the host receiver and compares all bytes with independently generated
reference output. This can take substantially longer than the core-only test.

Earlier PASS results for the full-trace core do not replace these new tests.

## Compare builds under the same installed toolchain

After the simulations pass:

```bash
bash build.sh baseline
bash build.sh
```

| Build | Output | Synthesis log | Placement/routing log |
| --- | --- | --- | --- |
| Original baseline | `build/top_baseline.bit` | `build/baseline_synth.log` | `build/baseline_pnr.log` |
| Minimal stream | `build/top_stream.bit` | `build/stream_synth.log` | `build/stream_pnr.log` |

If baseline succeeds and stream fails, that is direct evidence that adding this
instrumentation changes the outcome under the current toolchain. If baseline
also fails, investigate toolchain/source reproducibility before attributing the
failure to the stream interface. Resource counts and physical legality still
need inspection; they are not predicted by the source diff alone.

Placement/routing has a five-minute wall-clock limit. A stop at that limit prints
an explicit timeout explanation and exits with status 124; it does not mean the
design was proved illegal or impossible to fit. Override only when justified:

```bash
PNR_TIMEOUT=15m bash build.sh
```

Tool paths can be overridden with `NEXTPNR`, `CHIPDB`, `FPGA_AS`, and `XRAY_DB`.
No placer workaround, carry disabling or legality bypass is enabled by default.
The build completes assembly only after nextpnr succeeds. A `.pending.bit` left
after failure is not a successful build. Source/bitstream hashes are written to
`build/stream_build_manifest.json` or `build/baseline_build_manifest.json`.

## Program and receive

After the stream build exits successfully, program from WSL:

```bash
openFPGALoader -b cmod_s7 build/top_stream.bit
```

Return the USB device to Windows with your working usbipd detach procedure,
keeping the board powered. Press/release button 0, then from this new directory
in Windows run:

```powershell
python -m pip install -r requirements.txt
python host/run_stream.py --port COM7
```

Use the actual UART port if COM7 changes. Default files are the existing demo
weights/config/input. `--weights`, `--config`, `--input`, and `--event` override
them. Expect roughly 20 seconds including parameter/input upload, streamed
outputs, computation and input replay, rather than compute-only latency.

The receiver checks its distinct `B5 01` identity. Use `run_stream.py` with this
bitstream; the earlier `run_trace.py` expects a different protocol. Sending the
identity command to the original bitstream requires resetting it before retrying
with the original sender.

## Results

Every invocation creates a new timestamped directory under `stream_results`:

- `report.json`: per-layer mismatch counts, first CHW coordinate, reference and
  local recomputation results, score consistency and original input replay check.
- `stem.npy`, `conv1.npy`, `conv2.npy`, `conv3.npy`, `pool.npy`: produced FPGA values.
- `input_replay.npy`: the original input returned at the end.
- `rx.bin`: incoming UART bytes, for offline analysis.
- `host_manifest.json`: hashes of the input/weight/config files used.

Send `report.json` first. `completed: true` means collection finished, not that
all numbers matched. Numerical mismatches are recorded and collection continues.
An interrupted/truncated protocol exchange needs button-0 reset before retrying.

Each layer is compared against the original complete NumPy inference, and also
against NumPy recomputed from the preceding **emitted** map and the **sent**
parameters. Unlike the full trace, the actual loaded weights are not independently
verified. The second check can distinguish propagated differences from a new
boundary discrepancy, but includes possible storage/parameter/transfer faults.
The pooled-vector check averages the emitted conv3 values, preserving SV's signed
division convention if an invalid negative map appears.

## Continuous mode and original sender compatibility

Press/release button 0 before a fresh session:

```powershell
python host/run_stream.py --port COM7 --continuous
```

This uses the same minimal-stream bitstream with no word stalls or intermediate
bytes. The original continuous state sequence and original `I` response are
retained. Consequently the included original `host/run_cnn.py` also works:

```powershell
python host/run_cnn.py --port COM7 --weights demo/demo_weights.npz --config demo/demo_config.json --input demo/demo_input.npy
```

Instrumentation changes placement; streaming additionally changes execution
timing. If an original hardware discrepancy disappears, neither observation
alone establishes that its cause was fixed. Keep the original bitstream/logs.

Saved captures can be analyzed offline with the same intended data files:

```powershell
python host/run_stream.py --capture stream_results/RUN_DIRECTORY/rx.bin
```

Add `--continuous` for a continuous capture. Explicit `--output` paths must be
new/empty to avoid mixing results from different runs.

## Minimal protocol

- `H`: return bytes `B5 01` (minimal-stream identity/version).
- `W`: unchanged ten configuration bytes + 15,105 little-endian int16 words;
  unchanged one-byte acknowledgment `A1`.
- `I`: unchanged 4,992-word input and continuous response.
- `T`: same input, streamed produced layer words, then the unchanged final response.

The `T` response is:

| Tag | Following little-endian int16 words |
| --- | ---: |
| `C0` | Stem: 39,936 |
| `C1` | Conv1: 19,968 |
| `C2` | Conv2: 9,984 |
| `C3` | Conv3: 2,560 |
| `C4` | Pool: 32 |

All layer words are CHW, last axis fastest. After them comes the original response:
`D1`, signed int16 score, 9,984 original input bytes, `D2`.
The host knows fixed counts; it does not scan tensor bytes for tag patterns.
There is no layer CRC or generic packet-length datapath in this minimal variant.
Do not transmit new commands while inference/output is active.

LEDs retain their meanings: heartbeat, weights accepted, sticky protocol error,
engine busy. In streaming mode, busy includes waiting for UART acceptance.
