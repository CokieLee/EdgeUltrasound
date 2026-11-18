# Validation status

Executed here:

- Three NumPy reference tests: passed. They exercise convolution against separate
  scalar loops, signed rounding/saturation/ReLU, parameter file layout and complete
  network dimensions.
- Synthetic full-network reference: score -50.
- Reference .mem fixtures generated for all intermediate arrays, pooling, final
  output and preserved input.
- Python compilation and shell syntax checks: passed.

Not executed here (tools/board unavailable):

- Icarus RTL simulation or elaboration.
- Yosys synthesis, BRAM/DSP utilization, nextpnr routing and timing.
- Full UART command/response protocol simulation.
- Physical FPGA transfer/inference and replay.
- Comparison with the user's trained/quantized network: actual weights and
  quantization script have not been supplied.

The previously tested UART RX/TX modules are reused, but that does not establish
correctness of the newly written protocol controller or CNN engine. Run the
included RTL tests before synthesis, and test the synthetic model on-board before
using actual inference results.
