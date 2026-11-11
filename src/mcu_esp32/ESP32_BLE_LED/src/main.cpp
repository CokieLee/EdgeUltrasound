#include <Arduino.h>
#include "model_data.h" // Your mock model header

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// 1. Setup Tensor Arena (Memory scratchpad)
constexpr int kTensorArenaSize = 30 * 1024; // 30 KB
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

// TFLM Globals
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_tensor = nullptr;
TfLiteTensor* output_tensor = nullptr;

// Constants matching your data dimensions
const int NUM_CHANNELS = 32;
const int NUM_SAMPLES = 256;
int16_t streamingBuffer[NUM_CHANNELS][NUM_SAMPLES]; 

void setup() {
  Serial.begin(500000);
  while(!Serial);

  // 2. Load the mock model from the C array
  model = tflite::GetModel(dummy_model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    return;
  }

  // 3. Register operations used by the model
  static tflite::AllOpsResolver op_resolver; // Convenient for testing (or use MicroMutableOpResolver)

  // 4. Instantiate the interpreter
  static tflite::MicroInterpreter static_interpreter(
      model, op_resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // 5. Allocate tensors
  interpreter->AllocateTensors();

  // 6. Grab pointers
  input_tensor = interpreter->input(0);
  output_tensor = interpreter->output(0);
  
  Serial.println("TFLM Initialized successfully with mock weights!");
}

void loop() {
  int total_bytes = NUM_CHANNELS * NUM_SAMPLES * sizeof(int16_t);
  
  // Check if your python streaming script sent data over serial
  if (Serial.available() >= total_bytes) {
    Serial.readBytes((char*)streamingBuffer, total_bytes);

    // --- PRINT CONFIRMATION UPON COMPLETE RECEPTION ---
    Serial.println("Full frame received completely! Running inference...");

    // Copy incoming stream data into the TFLM input tensor 
    // (casting/mapping dummy or real incoming bytes)
    for (int i = 0; i < (NUM_CHANNELS * NUM_SAMPLES); i++) {
      input_tensor->data.f[i] = (float)((int16_t*)streamingBuffer)[i]; 
    }

    // Run Inference
    int64_t start_time = esp_timer_get_time();
    TfLiteStatus status = interpreter->Invoke();
    int64_t duration = esp_timer_get_time() - start_time;

    if (status != kTfLiteOk) {
      Serial.println("Inference failed!");
      return;
    }

    // Print timing for your future ESP32 vs. FPGA benchmarking
    Serial.print("Mock Inference execution time (us): ");
    Serial.println(duration);

    // Acknowledge back to PC script
    Serial.write(0xAA);
  }
}