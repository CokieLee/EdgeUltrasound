import datetime
import numpy as np
import tensorflow as tf
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_MODEL = OUTPUT_DIR / "dummy_model_int16.tflite"
OUTPUT_HEADER = OUTPUT_DIR / "model_data.h"

# 1. Build a dummy model matching your input dimensions (256 time points, 32 channels)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(256, 32, 1)),
    tf.keras.layers.Conv2D(4, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(2, activation='softmax'),
])

# 2. Configure for INT16 Activations & INT8 Weights Quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Use the correct experimental ops set matching your Python version's attribute name
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
]

# Representative dataset for range calibration
def representative_dataset_gen():
  for _ in range(10):
    data = np.random.uniform(-1.0, 1.0, size=(1, 256, 32, 1)).astype(np.float32)
    yield [data]

converter.representative_dataset = representative_dataset_gen

# Enforce INT16 inputs and outputs to match your sensor stream
converter.inference_input_type = tf.int16
converter.inference_output_type = tf.int16

tflite_model = converter.convert()

# 3. Write out to a binary file
with OUTPUT_MODEL.open("wb") as f:
  f.write(tflite_model)

# 4. Programmatically generate the C header file
OUTPUT_HEADER.parent.mkdir(parents=True, exist_ok=True)
byte_values = ", ".join(f"0x{byte:02x}" for byte in tflite_model)
with OUTPUT_HEADER.open("w", encoding="ascii") as f:
  f.write("#ifndef MODEL_DATA_H\n#define MODEL_DATA_H\n\n")
  
  # --- METADATA COMMENTS ---
  f.write("// ========================================================\n")
  f.write("// AUTO-GENERATED TFLITE MODEL HEADER\n")
  f.write(f"// Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
  f.write("// Format: INT16 (Activations) / INT8 (Weights)\n")
  f.write("// Input Shape: [1, 256, 32, 1] (256 samples x 32 channels)\n")
  f.write(f"// Model Size: {len(tflite_model)} bytes\n")
  f.write("// ========================================================\n\n")
  
  # --- ARRAY DATA ---
  f.write(
      f"const unsigned char dummy_model_tflite[] = {{{byte_values}}};\n"
  )
  f.write(f"const int dummy_model_tflite_len = {len(tflite_model)};\n\n")
  f.write("#endif // MODEL_DATA_H\n")

print(f"Successfully generated annotated INT16 model_data.h at {OUTPUT_HEADER}")
print(f"Successfully generated C header at {OUTPUT_HEADER}")