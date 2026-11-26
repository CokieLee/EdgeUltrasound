# Streaming SNN V2 for the Cmod S7-25

This port uses the supplied trained `skip8` checkpoint. The 665 parameters are built into the FPGA as two byte-wide weight ROMs. Programming the bitstream loads the model. The UART host checks the ROM fingerprint before sending tokens.

| | Cmod S7 SNN V1 | Streaming SNN V2 |
| --- | --- | --- |
| Input | One 32 × 156 RF channel reused on four steps | Sixteen distinct 64-feature RF tokens in time order |
| Layers | Convolutional stem and three spiking convolutions | Two eight-neuron dense LIF layers and a linear skip branch |
| LIF | Beta ≈ 0.9, strict threshold 1, delayed subtract reset | Beta = 7/8, inclusive threshold 0.5, immediate subtract reset |
| Readout | Count final-layer spikes over space and time | Mean pre-reset layer-2 voltage, plus a linear skip of the mean input |
| Model storage | Upload weights over UART | Weights initialized in the bitstream |

## Build and program

Run in WSL from the extracted `cmod_s7_snn_v2` directory with the same Yosys/OpenXC7 setup as V1.

```bash
python3 -m pip install -r requirements.txt
python3 sim/test_host.py
bash sim/run.sh
bash build.sh
openFPGALoader -b cmod_s7 build/top_snn_v2.bit
```

`sim/run.sh` uses Icarus Verilog. It drives the actual UART RTL and compares all 2,220 response bytes across three requests: traced demo, the same demo in score-only mode, and a traced stress fixture including both full-scale and zero-valued inputs. The requests run without a board reset between frames. `build.sh` uses the V1 board pin map and 12 MHz clock, checks the synthesized weight ROMs for parity data use, and records synthesis and routing logs in `build/`.

## Run on Windows

Return the USB device to Windows, check which COM port enumerated, and run:

```powershell
python -m pip install -r requirements.txt
python host/run_snn_v2.py --port COM7 --trace
python host/run_snn_v2.py --port COM7
```

Both commands run one frame. The default fixture supplies synthetic input tokens to the trained checkpoint. The reference result is signed Q20 integer **-1055882**, or standardized score **-1.00696754**. The float baseline is **-1.00705040**, with identical spike decisions on the fixture. The host also checks a CRC of the received FPGA input and, with `--trace`, checks both layers' spikes and pre-reset voltages for each token. Use `--output report.json` to save the comparison. The LEDs show heartbeat, frame active, protocol error and compute busy, in index order. The busy bursts are short enough to be difficult to see.

For real inputs, pass one of the following. Each accepts a NumPy `.npy` file.

| Argument | Expected data |
| --- | --- |
| `--rf one_frame.npy` | Original unnormalized `[384,64,250]` RF frame. The host selects the same eight events as the supplied model, pads 250 samples to 256, and builds mean-absolute-amplitude tokens. |
| `--tokens encoded.npy` | Encoded `[16,64]` nonnegative amplitudes before gain and clipping. |
| `--normalized normalized.npy` | Already normalized `[16,64]` values in `[0,5]`. |

```powershell
python host/run_snn_v2.py --port COM7 --rf one_frame.npy --trace --output report.json
python host/run_snn_v2.py --reference-only --rf one_frame.npy
```

The host uses the checkpoint's gain **81.0234375** and clips normalized tokens to `[0,5]`. It converts the standardized FPGA score with `17.43912124633789 + 10 * score`, as in the supplied model. To reproduce a result from another device, use the same normalized tokens and checkpoint on both.

## Datapath and arithmetic

- Buffer one 64-feature token and send a UART credit after processing it. The FPGA carries 16 membrane states across the frame and clears them at the start of the next frame.
- Compute fc1, fc2, and the skip dot product for each token. Hardware shares a signed multiply-accumulate datapath across fc1, skip and the head. In fc2 each spike gates a signed weight. Bias is applied every token in both LIF layers.
- Compute `v = (7*m >> 3) + current`, `spike = (v >= 0.5)`, then `m = v - 0.5*spike`. Accumulate **v2** before subtracting the reset. At the last token, combine the head and skip contributions and divide by 16 using a shift. Linearity allows the skip dot product to be summed across tokens before dividing.
- Input is unsigned Q12, weights and biases are signed int16 with 15 fractional bits, voltages use signed Q18 and the returned score is signed Q20. LIF decay uses an arithmetic shift. The first-layer current and final score round to nearest with ties toward positive infinity. The fixed-point host comparison must be bit-exact. Float equality can vary near thresholds.
- The schedule uses roughly 21,000 FPGA compute cycles per frame, around 1.8 ms at 12 MHz before UART time. The 2 KiB input alone takes roughly 178 ms on a 115200-baud 8N1 link. These are estimates from the schedule and wire rate.

## Prepared checkpoint

`weights/lo.mem`, `weights/hi.mem` and `weights/model_id.svh` are consumed by simulation and synthesis. `weights/reference.npz` contains the float and fixed weight tensors. `weights/config.json` records scales, model metadata, and hashes. Re-export this checkpoint's files with:

```bash
python3 host/export_weights.py weights/source.pt weights
```

The ROM is row-major in this order:

| Word offset | Tensor | Count |
| ---: | --- | ---: |
| 0 | `fc1.weight` | 512 |
| 512 | `fc1.bias` | 8 |
| 520 | `fc2.weight` | 64 |
| 584 | `fc2.bias` | 8 |
| 592 | `out.weight` | 8 |
| 600 | `out.bias` | 1 |
| 601 | `skip.weight` | 64 |

PyTorch is only needed if running the supplied original `host/trained_model.py`. The exporter understands the supplied CPU float checkpoint format and checks the expected shapes. The NumPy reference and UART host use the prepared files directly.

## Verification status

Python tests passed for the checkpoint, ROM bytes, RF layout and padding, fixed-point score and spikes, and the host packet parser. RTL simulation, synthesis, place-and-route, and the board run still need verification with the FPGA tools.
