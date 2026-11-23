# Cmod S7 sequential CNN, signed int16

An initial handwritten SystemVerilog implementation of the non-spiking reduced
RatClassifier. One event is processed per request. It uses the existing 12 MHz
clock, USB-UART pins and open-source FPGA build flow.

**Status: initial RTL, not yet hardware-validated.** See VALIDATION.md. Run the
simulation first. The supplied synthetic weights are NOT trained rat-liver weights.
Your actual quantized weights and scaling conventions must be supplied before
this can represent your trained classifier.

## Fixed network

| Stage | Output CHW | int16 storage |
| --- | --- | ---: |
| One input event | 1 x 32 x 156 | 9.75 KiB |
| Stem, no ReLU | 8 x 32 x 156 | 78 KiB |
| Conv1 + ReLU | 16 x 16 x 78 | 39 KiB |
| Conv2 + ReLU | 32 x 8 x 39 | 19.5 KiB |
| Conv3 + ReLU | 32 x 4 x 20 | 5 KiB |
| Global average | 32 | 64 bytes |
| Linear head | 1 | 16-bit score |

Convolutions 1..3 use 3x3 kernels, stride 2, padding 1. The stem is 1x1,
stride 1, padding 0. The engine executes PyTorch-style cross-correlation with
weight order [output channel, input channel, kernel row, kernel column].
For the model's default channel_reduction=1, use receive event 0; no need to send
the 383 events the model ignores. Selecting another event changes the input to
the trained model and should be evaluated deliberately.

The engine uses one sequential MAC operation at a time, 48-bit signed
accumulation, synchronous memory reads and a state-machine scheduler. It does
not attempt parallel acceleration, layer fusion or sparse computation. Resource
mapping and achievable timing still need synthesis and placement/routing checks.

## Memory budget

| Memory | Words x bits | Payload |
| --- | --- | ---: |
| Buffer A | 39,936 x 16 | 78 KiB |
| Buffer B | 19,968 x 16 | 39 KiB |
| Parameters | 15,105 x 16 | 29.502 KiB |
| Preserved original event | 4,992 x 16 | 9.75 KiB |
| Total main arrays | | 156.252 KiB |

The original event is retained specifically to return it after inference. The
stem reads it and writes A; conv1 reads A and writes B; conv2 reads B and writes A;
conv3 reads A and writes B. Pooling then reads B. The preserved input is also the
stem's source, so there is no second initial input copy in a work buffer.

At a nominal 2,048 int16 words per 36-Kibit RAM, these arrays require approximately
20 + 10 + 8 + 3 = 41 of 45 blocks. Small registers, packing, inference choices and
mapping can change utilization. This is NOT a verified fit report. No large
memory array is reset; only controls reset, and inputs/weights are loaded before use.

## Exact numerical contract (must match your quantization script)

Inputs, all kernel/linear weights, biases and stored layer outputs are signed int16.
Weights are runtime-loaded, not synthesized as constant arithmetic.

For each layer l, including the head:

```
acc = sign_extend_48(bias_int16) << bias_left_shifts[l]
acc += sum(input_int16 * weight_int16)  # exact signed 16x16 products
q = arithmetic_right_shift(acc, right_shifts[l])
output = saturate_int16(q)             # stem and head
output = saturate_to_0_32767(q)         # conv1, conv2, conv3 (ReLU)
```

Arithmetic right shift rounds negative values toward negative infinity, NOT to
nearest and NOT toward zero. Activation outputs are requantized/saturated after
every layer. Pooling uses floor(sum / 80), with no extra scale change; its inputs
are nonnegative ReLU outputs. The head also returns a saturated int16 value, NOT
an IEEE float or automatically calibrated probability.

The two lists each have five integers 0..31, in order stem, conv1, conv2, conv3,
head. Only symmetric quantization (zero point 0), per-layer power-of-two scaling,
and the rounding above are directly supported. Arbitrary multipliers, per-output-
channel scales, nearest rounding and int32 biases require changes. Do not silently
cast your existing quantized model into this contract.

For binary fixed-point scales F_in/F_weight/F_out (fractional bit counts), a
candidate shift is F_in + F_weight - F_out. If a bias has F_bias fractional bits,
its left shift is F_in + F_weight - F_bias. These must be nonnegative within the
supported limits, and precision/range must be evaluated against the trained model.
There is no claim of FP32 equivalence or retained accuracy without that evaluation.

## Files and model export

`host/cnn_reference.py` implements the exact integer arithmetic above and validates
the export format. Save a NumPy .npz with these signed int16 arrays:

| Keys | Weight shape | Bias shape |
| --- | --- | --- |
| stem.weight, stem.bias | (8,1,1,1) | (8,) |
| conv1.weight, conv1.bias | (16,8,3,3) | (16,) |
| conv2.weight, conv2.bias | (32,16,3,3) | (32,) |
| conv3.weight, conv3.bias | (32,32,3,3) | (32,) |
| head.weight, head.bias | (1,32) | (1,) |

For example, if your quantization script ALREADY produces these int16 arrays in
a dictionary `quantized_arrays`, export them with:

```python
np.savez("weights_int16.npz", **quantized_arrays)
```

Do not export the original FP32 state_dict by simply casting it. Adapt/verify
scaling using your quantization script. A separate JSON configuration contains:

```json
{
  "right_shifts": [0, 0, 0, 0, 0],
  "bias_left_shifts": [0, 0, 0, 0, 0]
}
```

Those zeros illustrate the schema; they are NOT calibrated settings for your
model. Use values appropriate to your fixed-point model. Input .npy files must
already be int16, with shape [1, events, 32, 156] or [events, 32, 156]. The sender
selects one event, without changing its numeric values. Output dequantization
uses the output scale from your quantization script.

## 1. Test the integer reference and RTL

From this project directory in WSL:

```bash
python3 -m unittest discover -s host -p 'test_*.py' -v
bash sim/run.sh
```

The RTL testbench loads a generated signed synthetic model and input, checks EVERY
stored output from the stem and three convolutions against the NumPy reference,
checks pooling and the final scalar, and verifies the original input is preserved.
It also elaborates the board wrapper. The simulation covers millions of clocks,
so it can take appreciably longer than the UART test. It does not yet simulate
the entire UART command protocol; that needs a board test or a protocol testbench.
The 10 ns testbench period only controls simulated time; counters execute the same
cycle sequence as the board. Simulation is not a physical timing check.

## 2. Synthesize separately and inspect utilization

```bash
mkdir -p build
yosys -l build/synth.log -p 'read_verilog -sv rtl/uart_rx.sv rtl/uart_tx.sv rtl/cnn_engine.sv rtl/top.sv; synth_xilinx -family xc7 -top top; check -assert; stat; write_json build/top.json'
```

Check inferred block RAM and logic counts, then proceed only if sensible. To run
the complete known-path synthesis / route / fpga-as flow:

```bash
bash build.sh
```

The script uses the tool and database paths from your working UART project,
including `$HOME/prjxray-db/spartan7` for fpga-as. Source your usual OSS CAD
environment if yosys is not on PATH. A successful HLS-free build is still not
guaranteed until this code has been synthesized and routed on your installation.

## 3. Upload and reset

```bash
openFPGALoader -b cmod_s7 build/top.bit
```

Press and release button 0 after upload. This resets controller state but does not
erase every BRAM cell; the protocol requires a complete weight and input load.

| LED | Meaning |
| --- | --- |
| 0 | heartbeat |
| 1 | parameter/config load accepted |
| 2 | sticky serial/protocol error: reset before retrying |
| 3 | inference engine busy |

This is a new top and protocol. The old echo-only `send_tensor.py` does NOT work
with this bitstream. Conversely, the new sender does NOT work with the old UART
echo bitstream.

## 4. Initial board smoke test (synthetic weights)

If programming in WSL but sending from Windows, detach the FTDI device from WSL
as before, keeping the cable/power connected. The known Windows port was COM7;
check it again if Windows assigns another number.

In the PC sender's Python environment:

```bash
python -m pip install -r requirements.txt
python host/cnn_reference.py --make-demo demo
python host/run_cnn.py --port COM7 --weights demo/demo_weights.npz --config demo/demo_config.json --input demo/demo_input.npy
```

The synthetic model is deliberately untrained. It should produce integer score
**-50** with the deterministic generated input. The sender also compares every
returned input byte. This tests execution and transport, not prediction quality.

For your trained model, replace the three files with your int16 weight archive,
calibrated configuration and int16 input .npy. Use `--event 0` (the default) to match
the supplied model's event selection. Start with one event and compare the RTL
against the integer reference before evaluating prediction accuracy.

## UART protocol, 115200 baud / 8N1

1. PC sends byte `0x57` ('W'), then ten configuration bytes interleaving right/bias
   shifts for layers 0..4, then 15,105 signed int16 parameters, little-endian.
   Flatten each layer's weight array in C/PyTorch order, followed by its biases,
   then the next layer. FPGA replies `0xA1` after the load. There is no full weight
   echo, CRC, or readback yet: this acknowledgement confirms length acceptance,
   not weight integrity. End-to-end score comparison is still required.
2. PC sends `0x49` ('I'), then 4,992 signed int16 input samples, little-endian,
   in channel/row/sample order (last axis fastest). No immediate echo is sent.
3. FPGA runs the CNN. PC must not send another command while inference/replay is
   active. Parameters and input remain valid until replaced or reset.
4. FPGA sends `0xD1`, little-endian int16 score, all original 9,984 input bytes,
   then `0xD2`. This is the requested delayed input replay, not a feature-map dump.

No header/packet timeout, CRC, retries or RTS/CTS is implemented. A partial upload
requires resetting and reloading; use only one sender. The host has timeouts and
checks markers, score and exact replay. The default response timeout is 60 seconds.

Weight upload is about 2.6 s and each input/replay direction about 0.87 s minimum
at 115200 baud. The host's reported elapsed interval includes transfer and inference;
do not report it as compute-only latency. This is a correctness-first prototype.

## First errors to share

Start with `bash sim/run.sh` and share its first failing layer/index or PASS lines.
Then share synthesis utilization, especially BRAM counts. Do not treat the memory
budget as proof of actual resource fit, or the integer smoke test as evidence of
trained-model accuracy.
