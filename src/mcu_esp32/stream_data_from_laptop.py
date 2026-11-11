import time
import serial

# ==========================================
# CONFIGURATION CONSTANTS (Scale these later!)
# ==========================================
NUM_CHANNELS = 1
NUM_SAMPLES = 4
DATA_TYPE_SIZE = 2  # 2 bytes for int16_t
# will eventually do 32 channels with 256 samples per channel...

# Serial configuration (Adjust COM port and baud rate as needed)
PORT = "COM11"
BAUD_RATE = 500000

# Calculate total payload size in bytes
TOTAL_BYTES = NUM_CHANNELS * NUM_SAMPLES * DATA_TYPE_SIZE

import numpy as np

# Open connection
ser = serial.Serial(PORT, BAUD_RATE, timeout=1)
time.sleep(2)  # Allow connection to settle

print(
    f"Streaming {NUM_CHANNELS} channel(s) with {NUM_SAMPLES} samples"
    f" ({TOTAL_BYTES} bytes per frame)..."
)

try:
  while True:
    # 1. Generate a test frame matching our dimensions and int16 type
    # For testing, we use predictable values (e.g., counting up or random integers)
    frame = np.arange(
        NUM_CHANNELS * NUM_SAMPLES, dtype=np.int16
    ).reshape(  # or np.random.randint(-100, 100, size=(NUM_CHANNELS, NUM_SAMPLES), dtype=np.int16)
        NUM_CHANNELS, NUM_SAMPLES
    )

    # 2. Send the raw bytes over serial
    ser.write(frame.tobytes())

    # 3. Wait for acknowledgement byte from the ESP32
    ack = ser.read(1)
    if ack:
      print(f"Sent frame successfully. Data: {frame.tolist()}")
    else:
      print("Warning: No acknowledgement received from ESP32.")

    time.sleep(0.5)  # Pause between frames for easy viewing

except KeyboardInterrupt:
  print("\nStreaming stopped by user.")
  ser.close()