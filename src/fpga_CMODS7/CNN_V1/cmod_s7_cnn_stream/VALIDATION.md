# Validation status

Executed here:

- 11 Python tests passed: three reference tests and eight stream-receiver tests.
- Correct full and continuous captures passed; injected convolution/pooling
  faults, invalid tags, truncation, wrong final markers, input replay corruption,
  partial I/O and signed comparisons were checked.
- An offline Python-generated full capture (154,956 device bytes) was received,
  saved and compared successfully. This was not an HDL or hardware capture.
- Shell syntax and Python compilation checks passed.
- The original RAM read/write always block is byte-for-byte identical in the
  stream core. Baseline core/top are byte-for-byte copies of the original
  workspace files. Engine and top diffs are included for review.

Not executed here (HDL/toolchain/board unavailable):

- New stream core elaboration/simulation, or pin-level UART simulation.
- Synthesis, actual resource utilization, placement, routing, timing or assembly.
- Baseline versus stream physical build comparison with the user's toolchain.
- Board behavior, emitted tensors, or whether the original -73 reproduces.

Run the included simulations on your machine before building. A smaller source
change does not guarantee a legal physical implementation. These outputs tap
produced values and do not independently verify stored RAM or loaded parameters.
