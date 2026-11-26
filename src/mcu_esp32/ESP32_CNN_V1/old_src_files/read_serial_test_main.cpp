#include <Arduino.h>

const int NUM_CHANNELS = 1;
const int NUM_SAMPLES = 4;
const int TOTAL_BYTES = NUM_CHANNELS * NUM_SAMPLES * sizeof(int16_t);

int16_t dataBuffer[NUM_CHANNELS][NUM_SAMPLES];

void setup() {
  Serial.begin(500000);
  while(!Serial);
}

void loop() {
  if (Serial.available() >= TOTAL_BYTES) {
    Serial.readBytes((char*)dataBuffer, TOTAL_BYTES);

    // Print out received values to Serial Monitor to verify
    Serial.print("Received Sample: ");
    Serial.println(dataBuffer[0][0]); // Print first value of the 4 samples

    // Acknowledge back to PC
    Serial.write(0xAA);
  }
}