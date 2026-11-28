# ESP32 SNN V2.1 preprocessing power benchmark

Make a copy of your working `ESP32_SNN_V2.1_powereval` project named `ESP32_SNN_V2.1_preprocess_powereval`. Overlay the archive at the project root. Keep the working PlatformIO config and choose the same CPU frequency when comparing to SNN-only. If an older `snn_model_data.h` sits beside `src/main.cpp`, replace it with the packaged header there too. It must not shadow the V2.1 header with different weights.

## Run

In the new project, using its assigned COM port:

```powershell
pio run -t upload --upload-port COM11
python tools/send_snn.py COM11 --power-benchmark 30
```

The default file is `C:\Users\ethan\Documents\Duke\Classes\Neuromorphic\Stanford_Murine_Processed_RFData_SNNV2\skip8_V2.1_1127\unprocessed\rat01_frame00.npy`. It must hold original-amplitude signed int16. To choose a different file or a directory, pass its path after the port:

```powershell
python tools/send_snn.py COM11 "C:\path\to\unprocessed" --power-benchmark 30
```

A directory runs each `.npy` in filename order. Use a single frame for taking and recording a current reading. The sender appends to `mcu_preprocess_power_v2.1.json` by default. Optionally supply `--voltage`, `--current-ma` and `--idle-current-ma` to calculate power and energy from a prior comparable reading.

## What is measured

- Python keeps every second channel and fourth sample from `[384,128,1000]`, and selects events `0,54,109,164,218,273,328,383`. Existing `[384,64,250]` files skip channel/sample striding. An already selected `[8,64,250]` file must be in the same event order. The payload contains only those 128,000 int16 samples in event/channel/sample order (256,000 bytes).
- The host loads one frame into the ESP32, in acknowledged chunks. The ESP32 keeps the data in eight internal RAM allocations of 32,000 bytes each. It reports an allocation error if the firmware cannot reserve all eight. There is no PSRAM or flash dependency for the frame.
- During the timed loop, `preprocess.cpp` pools absolute amplitudes across eight channels and 16 samples per bin. It treats samples 250–255 as zero and still divides by 128 for the final bin. It writes 16 rows with 64 event-by-channel-group features. The existing inference performs gain division (81.0234375), clipping to [0,5], and the V2.1 float32 SNN. The timer covers these operations on every repeat, plus the same timer checks and brief scheduler yields as the SNN-only benchmark.
- The serial upload, RF CRC, final pooled-token CRC, and output printing are outside the timed loop. Python independently computes the expected pre-gain token CRC for validation. Timing excludes acquisition and host-side selection. Label this **pooled RF preprocessing + SNN with reduced acquisition** in your report.

After the upload and two-second countdown, take the mean current during the window. The ESP32 reports actual CPU MHz, count, elapsed time, score, and the CRCs. Desktop tests with the ESP32 interfaces stubbed verified the raw and reduced paths, token pooling including -32768 and the final padded bin, the score, and appending consecutive results. ESP32 compilation, internal RAM fit, and board measurements remain to be verified.
