# Validation status

Actually run while preparing this package:

- `python3 -m unittest discover -s host -p 'test_*.py' -v`: all six tests passed.
- `python3 -m py_compile host/send_tensor.py host/test_send_tensor.py`: passed.
- `python3 host/send_tensor.py --help`: passed.
- `bash -n build.sh` and `bash -n sim/run.sh`: passed.
- Pin assignments checked against Digilent's Cmod S7-25 master XDC.

Not run in the preparation environment:

- RTL simulation: Icarus Verilog/Verilator were not available.
- Yosys synthesis and nextpnr placement/routing: tool binaries were not available.
- Physical Cmod S7 upload or serial-port testing: board not connected.

The included RTL testbench must be run before hardware deployment. Do not
interpret the Python transport-test results as validation of UART RTL timing,
FPGA resource mapping, or board communication.

Suggested first on-device acceptance: reset the loaded design, observe heartbeat,
run the default 1024-byte host pattern test, require PASS and an unlit error LED,
then test a small .npy slice before a complete tensor.
