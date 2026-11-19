# Diagnose a repeatable CNN score mismatch

These tools work with the original CNN bitstream and its W/I UART protocol.
Do not rebuild the FPGA for this first round of diagnosis. Keep the same bitstream
that returned -73 so that we test the failing implementation.

Use the same demo weights/config/input that produced the mismatch; do not regenerate them between tests.

## What the original failure establishes

The original sender checks framing and every input replay byte before checking
the score. A score mismatch at its final comparison therefore means the input
replay check passed. It does not establish weight integrity: A1 acknowledges a
complete length, without a weight readback or checksum. It also does not establish
that the engine read the original input RAM correctly during computation.

The original `tb_cnn_engine.sv` loads parameters directly through the engine's
write ports and checks all convolution outputs, pooling, score, and input replay.
It does not exercise the top-level UART controller. RTL simulation also does not
validate the synthesized/routed implementation. Three identical scores support
a repeatable fault; they do not rule out timing or memory mapping problems.

## First run: seven known-output tests

Use Windows Python with the USB device detached from WSL. Stop other senders and
serial terminals. Press and release button 0, then run from your CNN directory:

```powershell
python host/debug_cnn.py --port COM7
```

The default paths are `demo/demo_weights.npz`, `demo/demo_config.json`, and
`demo/demo_input.npy`. Override them with `--weights`, `--config`, and `--input`
if necessary. Event 0 is the default.

Each test reloads all weights, all configuration shifts, and the original input.
Do not press reset between tests. A complete score mismatch is recorded and the
next test runs; a transport failure aborts because the controller may need reset.
Each transaction still has a 60-second response timeout. Expect roughly a minute
or two for all seven tests on this initial sequential design, depending on the
actual computation time. The script prints progress for every test.

| Test | Expected | Purpose |
| --- | ---: | --- |
| baseline | -50 for the supplied demo | Reproduce the original mismatch |
| head_bias_123 | 123 | Zero weights; positive final bias and score transport |
| head_bias_-123 | -123 | Negative final bias and signed score transport |
| constant_low_channel | 64 | Constant through center taps, low channel addresses, pooling/head |
| constant_high_channels_bias_shift | 64 | Last channel of each layer and stem bias left shift |
| signed_head_and_right_shifts | -2 | All five right shifts and negative head multiplication |
| input_center_path | Calculated from your input | Input-dependent convolutions at stride 2 through three layers |

The input-dependent path selects the original samples `[::8, ::8]`, applies ReLU,
and floors their average over 80 samples. The other diagnostic models have the
simple expected answers above, independently checked against the NumPy network.
None of these diagnostic models are trained classifiers.

The script saves `cnn_debug_report.json`, including partial results if a transfer
fails, source/data file hashes, configuration, and each expected/actual score.
Share that file or paste its results. These hashes identify the host inputs, not
the firmware loaded into the FPGA.

## Optional: measure the 32 pooled channels

Reset button 0 before starting a new run, then:

```powershell
python host/debug_cnn.py --port COM7 --probe-pool --report cnn_pool_report.json
```

This repeats the seven tests, then runs 32 more inferences with the original
convolution parameters. For each inference, only the head is replaced: one weight
is 1, all others and the head bias are 0, and its output shift is 0. The returned
score should then equal that selected pooled channel. This takes several minutes
and prints progress for each channel.

This is a measurement through the head, not a raw BRAM readback. It depends on
that head and repeated parameter uploads working. Interpret it with the simple
tests, rather than treating every mismatch as proof of a pooling bug.

The report also evaluates the original head in Python using the measured pool:

- Different pooled values that reconstruct -73 would support a fault before the
  final head, assuming valid probes and consistent convolution weights each run.
- All pooled values matching while the baseline differs would focus attention
  on original head parameters, scaling, accumulation, or their hardware execution.
- Bias or simple path failures should be investigated before drawing conclusions
  from the pooled probes.

## Further evidence if needed

Keep your original synthesis and route logs. Share the final utilization and
timing sections of `build/synth.log` and `build/pnr.log`, plus any build errors.
If only `top.pending.bit` exists despite using the supplied build script, inspect
why its successful-finalization step did not run. A working filename alone does
not identify which source revision the bitstream contains.

If these tests do not localize the fault, the next firmware diagnostics should
read back configuration/weights exactly and pause after each layer to dump its
output before buffer reuse. That permits comparison with simulation at the first
different element. These host-only tools do not implement those commands.

## Validation of these tools

Run without hardware:

```powershell
python -m unittest discover -s host -p "test_*.py" -v
python host/debug_cnn.py --reference-only
```

Executed during preparation: all nine Python tests passed, including the three
existing reference tests. The six new tests check simple model answers, all 32
reference pool probes, signed response decoding, partial reads/writes, rejected
acknowledgments, replay corruption, and invalid response termination.

No FPGA or HDL simulator was available during this diagnostic-tool preparation.
No RTL, bitstream, synthesis settings, or CNN arithmetic was changed. These tools
collect evidence; they do not claim to fix the -73 hardware result.
