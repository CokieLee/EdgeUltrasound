# Streaming RF SNN on ESP32-S3

This is an FP32 Arduino port of the uploaded width-8 StreamingSNN with its skip connection. The weight header already contains the supplied checkpoint, including gain `81.0234375` and target mean `17.43912124633789` stored as FP32. The sketch uses the Arduino and ESP32 timing APIs and has no TensorFlow Lite Micro dependency.

## Build and run

1. Use the same PlatformIO environment and USB or UART settings as your working ESP32-S3 sketch. In the project copy you want to run as the SNN, put `src/main.cpp` in `src/` in place of the CNN entry point and `snn_model_data.h`. Compile only one Arduino entry point. Keep your working `platformio.ini`.
2. Upload with the usual PlatformIO task or run `pio run -t upload` from that project. Close the serial monitor before running the sender.
3. From the extracted bundle, install the sender dependencies and send the built-in test. Replace `COM5` with your port, such as `/dev/ttyACM0` on Linux.

   ```sh
   python -m pip install numpy pyserial
   python tools/send_snn.py COM5
   ```

   The current checkpoint should give a standardized score of about **−0.82217** and a prediction of **9.21738%** for the synthetic test input. Inference time is measured on the board.

## Run an RF frame or a token tensor

```sh
python tools/send_snn.py COM5 frame.npy
python tools/send_snn.py COM5 tokens.npy
```

- An RF file contains one original, unnormalized frame of shape `[384, 64, 250]`. The host selects eight events, pads six samples, and computes the same magnitude averages as `encode_rf` in the Python model.
- A token file contains one unnormalized `[16, 64]` array in time-major order, as returned by `encode_rf`. On the board, each token is divided by the checkpoint gain and clamped to `[0, 5]`. Do not divide by the gain or by a per-frame maximum on the host.
- The board computes `target_mean + 10 * score` once per frame. For rat-level output, average the ten frame predictions for that rat, as described in the Python model.

## Serial format and timing

Use the same serial port as your Arduino sketch at 500000 baud. Each request is exactly 4096 bytes of little-endian IEEE-754 FP32 tokens, in row-major order. Send one request at a time. The reply is an ASCII line `SNN_RESULT <score> <prediction_pct> <inference_us>` followed by a single byte `0xAA`. The receiver drains data as it arrives, so it can receive requests larger than the serial buffer. The request has no framing marker. If a transfer is interrupted, reset the ESP32 before retrying.

The timer covers on-device normalization, the 16 recurrent steps and readout, and output calibration. It excludes host RF encoding, token transfer, and serial printing.

## Export another compatible checkpoint

With PyTorch installed on the host, regenerate the header from a StreamingSNN checkpoint that stores `state_dict`, `gain`, and `target_mean`.

```sh
python tools/export_weights.py "weights(1).pt" src/snn_model_data.h
```

Copy the regenerated header into your PlatformIO project and rebuild.
