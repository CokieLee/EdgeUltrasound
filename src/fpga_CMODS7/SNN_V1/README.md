# Four-step SNN for the Cmod S7

The first FPGA target uses one preprocessed RF channel (32 × 156), the widths from `downsize_factor=2` (8, 16, 32, 32), and four timesteps. It targets the XC7S25 with the 12 MHz clock and the same 115200-baud UART as the CNN. The supplied demo contains synthetic parameters. A trained checkpoint must have a one-channel stem and matching layer shapes.

## Architecture

| Stage | Working CNN | SNN |
| --- | --- | --- |
| Stem | One signed 1 × 1 convolution | Same geometry and signed output. Store each word in two 8-bit data banks at the same address. |
| Conv1 | Convolution followed by ReLU | Compute each neuron’s current once. Apply that current to its LIF neuron for four steps. |
| Conv2 and conv3 | Convolution of dense 16-bit activations, followed by ReLU | At each step, select a signed weight or zero using the corresponding input spike bit. Add bias and advance the LIF state. |
| Intermediate storage | Dense feature maps | Four spike bits per neuron for conv1 and conv2. Reuse one membrane register while computing each neuron’s four steps. |
| Pool and head | Spatial mean per channel, then a linear head | Count conv3 spikes over space and time per channel. Apply the linear head to these counts and divide its biased numerator by 320. |
| UART | Continuous score or streamed feature maps | Continuous score or streamed stem, packed spike maps and counts. Both modes replay the input. |

The scheduling order is layer → neuron → timestep. Inside each layer, a neuron depends on its own previous membrane and on spikes produced by the previous layer at the same timestep. All four spike bits for the preceding layer are available before the next layer begins. The four steps use the same preprocessed RF frame, just as in the training model. There is no feedback between neurons within a layer or from a higher layer.

The LIF uses a strict greater-than comparison and delayed subtract reset. With threshold 1024 in signed Q10 and beta quantized to 29491 / 32768, its update is:

```text
reset     = (previous > 1024)
membrane  = floor(previous * 29491 / 32768) + current - 1024 * reset
spike     = (membrane > 1024)
```

Each neuron starts at zero for each inference. The convolution currents saturate to signed 16 bits and the membrane has 32 bits. The spiking surrogate affects training gradients only. Arithmetic right shift floors the leak for negative values. Input quantization, saturation, beta approximation and rounding can change spikes near the threshold compared with float PyTorch.

For each output channel, the counter sums its spikes over 4 × 20 spatial positions and four timesteps. If `count[c]` is that sum, the head uses `floor((sum(weight[c] * count[c]) + 320*bias) / 320)`, followed by its configured right shift and saturation. The equivalent float operation is averaging the four `head(pool(spk3))` logits. Dividing only once preserves fractions that would otherwise be lost to per-channel integer rounding. For a signed result, the iterative divider implements floor rounding.

The FPGA uses one convolution datapath. Gating the terms by spike bits avoids the activation multiplication in conv2 and conv3. The control logic still visits each tap, so the design’s runtime does not depend on spike sparsity. The current schedule uses roughly 31 million compute cycles per frame before UART backpressure, around 2.6 seconds at 12 MHz. The BRAM banks use explicit byte or nibble arrays. The build audits inferred RAM data connections for parity-lane use.

## Simulate and build

In WSL, from this directory:

```sh
python3 -m pip install -r requirements.txt
python3 sim/test_host.py
bash sim/run.sh
bash sim/run.sh uart
bash build.sh
```

The default RTL simulation compares 72,480 trace words, exercises trace backpressure, and checks the demo score and input replay in two successive inferences. The optional UART simulation compares 154,956 captured bytes against the Python-generated packet stream. Both full simulations can take several minutes. They require Icarus Verilog. The build reuses the CNN’s Yosys, openXC7 nextpnr, chip database and FPGA assembler defaults. It logs synthesis to `build/synth.log` and place-and-route to `build/pnr.log`. Place-and-route defaults to 12 minutes. Override with `PNR_TIMEOUT` if needed.

After a successful build, program from WSL:

```sh
openFPGALoader -b cmod_s7 build/top_snn.bit
```

Return the USB device to Windows as before and run from PowerShell, using the COM port enumerated for the board:

```powershell
python -m pip install -r requirements.txt
python host/run_snn.py --port COM7 --output stream_results/snn_trace.json
python host/run_snn.py --port COM7 --continuous
```

The bitstream returns UART identity `B6 01` before parameter upload. The synthetic score is **-180** in both modes. The trace reports per-step spike totals. Trace words for a neuron hold the earliest step in bit 0 and the latest in bit 3. The host also reconstructs counts from conv3 spikes and recomputes the score from the emitted counts and the sent head.

## Export a trained model

With the supported PyTorch model and a single preprocessed batch available, export the five convolution/linear weight and bias pairs. Use the RF tensor after the original preprocessing pipeline. For a one-channel checkpoint, select the same RF channel used during training. The class uses `channel_reduction` from outside the shown constructor.

```python
assert model.spiking and model.steps == 4 and model.stem.in_channels == 1
for cell in (model.act1, model.act2, model.act3):
    assert cell.reset_mechanism == 'subtract' and cell.reset_delay
    assert abs(cell.beta.item() - 0.9) < 1e-6 and cell.threshold.item() == 1.0

layers = ('stem', 'conv1', 'conv2', 'conv3', 'head')
state = model.state_dict()
np.savez('float_weights.npz', **{
    f'{layer}.{part}': state[f'{layer}.{part}'].detach().cpu().numpy()
    for layer in layers for part in ('weight', 'bias')
})
np.save('float_input.npy', preprocessed_x.detach().cpu().numpy())
```

```powershell
python host/export_float.py --weights float_weights.npz --input float_input.npy --event 0 --output exported
python host/run_snn.py --port COM7 --weights exported/weights.npz --config exported/config.json --input exported/input.npy
```

The exporter outputs a single selected channel, so the second command uses the default event index zero in the exported file. It rejects incompatible shapes and out-of-range parameters or RF inputs. A checkpoint with a multi-channel stem needs a different FPGA stem and input protocol. Compare fixed-point and floating-point predictions on validation data to quantify the effect of quantization.

## Build status

The Python host checks and synthetic vectors have been verified. RTL simulation, synthesis, place-and-route and on-board validation require the FPGA tools and hardware.
