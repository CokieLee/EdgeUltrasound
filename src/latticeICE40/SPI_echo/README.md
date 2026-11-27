# UPduino onboard SPI and clock diagnostic

Attach the UPduino to WSL exactly as for normal programming. Activate the same Python environment with PyFtdi installed.

```bash
bash build.sh
iceprog build/probe.bin
python3 probe.py
```

This temporarily replaces the application bitstream in flash. It echoes the FTDI's MOSI data directly onto MISO with the flash deselected. That path works without the 12 MHz clock. After it sees 1024 rising edges at GPIO 20, it inverts the echo. Keep the board connected throughout the run.

| Probe result | Interpretation |
| --- | --- |
| Inverted echo | End-to-end MOSI/MISO path works. GPIO 20 saw clock edges. This test does not measure the clock frequency or verify the SNN's SPI byte decoder. |
| Identical echo | SPI data path works. Check the oscillator route to GPIO 20. The SNN will remain in reset without a clock. |
| All bytes `ff` or another pattern | No valid FPGA echo. Check that `iceprog` verified the probe flash image. Report the GPIO and three raw reads to help diagnose the flash wiring or SPI electrical path. |

On UPduino v3.1 route the board's `12M` output to package pin `20`. If OSC/R16 is already populated, reuse it. Otherwise unplug power and jumper the header pads marked `12M` and `20`. Keep the factory flash settings for R12/R13. This is the same pin map as the SNN SPI port.

To restore the application, run `iceprog build/top_upduino_spi.bin` from the **SNN V2 SPI project directory** afterward. The probe does not contain any SNN data.
