#include <Arduino.h>
#include "model_data.h" // Your auto-generated INT16 model header


// NOTE: Tensorflow lite was removed from arduino website.... should be mindful of this workaround by some dude who uploaded a final version of it online.
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// 1. Setup Tensor Arena (Memory scratchpad for TFLM operations)
// Adjust size if your partner's final model requires more memory scratch space
constexpr int kTensorArenaSize = 30 * 1024; 
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

// TFLM Globals
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_tensor = nullptr;
TfLiteTensor* output_tensor = nullptr;

// Constants matching your 32 channels × 256 samples stream format
const int NUM_CHANNELS = 32;
const int NUM_SAMPLES = 256;
int16_t streamingBuffer[NUM_CHANNELS][NUM_SAMPLES]; 

void setup() {
  Serial.begin(500000);
  while(!Serial);

  // 2. Load the model from the C array defined in model_data.h
  model = tflite::GetModel(dummy_model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    return;
  }

  // 3. Register operations used by the model
  static tflite::AllOpsResolver op_resolver; 

  // 4. Instantiate the interpreter
  static tflite::MicroInterpreter static_interpreter(
      model, op_resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // 5. Allocate tensors from the arena
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("Tensor allocation failed!");
    return;
  }

  // 6. Grab pointers to input and output tensors
  input_tensor = interpreter->input(0);
  output_tensor = interpreter->output(0);
  
  Serial.println("TFLM Initialized successfully! Waiting for data stream...");
}

void loop() {
  int total_bytes = NUM_CHANNELS * NUM_SAMPLES * sizeof(int16_t);
  
  // Check if your python streaming script sent a complete frame over serial
  if (Serial.available() >= total_bytes) {
    // Read the complete frame into memory
    Serial.readBytes((char*)streamingBuffer, total_bytes);

    // --- PRINT CONFIRMATION UPON COMPLETE RECEPTION ---
    Serial.println("Full frame received completely! Running inference...");

    // Copy incoming stream data directly into the INT16 input tensor
    int total_elements = NUM_CHANNELS * NUM_SAMPLES;
    for (int i = 0; i < total_elements; i++) {
      input_tensor->data.i16[i] = ((int16_t*)streamingBuffer)[i]; 
    }

    // Run CNN Inference
    int64_t start_time = esp_timer_get_time();
    TfLiteStatus invoke_status = interpreter->Invoke();
    int64_t duration = esp_timer_get_time() - start_time;

    if (invoke_status != kTfLiteOk) {
      Serial.println("Inference failed!");
      return;
    }

    // Print timing results for your benchmarking
    Serial.print("Inference execution time (us): ");
    Serial.println(duration);

    // Optional: Print output classification result
    // int8_t result_class = output_tensor->data.int8[0];

    // Send back acknowledgment byte to your PC script
    Serial.write(0xAA);
  }
}