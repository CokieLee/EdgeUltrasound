#include <Arduino.h>
#include <esp_timer.h>
#include <limits>

#include "snn_model_data.h"

namespace {

constexpr size_t kTimesteps = 16;
constexpr size_t kFeatures = 64;
constexpr float kLeak = 0.875f;
constexpr float kThreshold = 0.5f;
constexpr uint8_t kAck = 0xAA;

static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
              "The serial input requires IEEE-754 binary32");

float tokens[kTimesteps][kFeatures];
size_t received = 0;

template <size_t N>
float dot(const float (&weights)[N], const float (&values)[N]) {
  float total = 0.0f;
  for (size_t i = 0; i < N; ++i) {
    total += weights[i] * values[i];
  }
  return total;
}

float infer() {
  using namespace snn_model;
  float membrane1[kWidth]{};
  float membrane2[kWidth]{};
  float voltage_total[kWidth]{};
#if SNN_HAS_SKIP
  float token_total[kFeatures]{};
#endif

  for (const auto &token : tokens) {
    float normalized[kFeatures];
    for (size_t i = 0; i < kFeatures; ++i) {
      const float scaled = token[i] / kGain;
      normalized[i] = scaled < 0.0f ? 0.0f : (scaled > 5.0f ? 5.0f : scaled);
#if SNN_HAS_SKIP
      token_total[i] += normalized[i];
#endif
    }

    float spikes1[kWidth];
    for (size_t neuron = 0; neuron < kWidth; ++neuron) {
      const float current = dot(kFc1Weight[neuron], normalized) + kFc1Bias[neuron];
      const float voltage = kLeak * membrane1[neuron] + current;
      spikes1[neuron] = voltage >= kThreshold ? 1.0f : 0.0f;
      membrane1[neuron] = voltage - kThreshold * spikes1[neuron];
    }

    for (size_t neuron = 0; neuron < kWidth; ++neuron) {
      const float current = dot(kFc2Weight[neuron], spikes1) + kFc2Bias[neuron];
      const float voltage = kLeak * membrane2[neuron] + current;
      membrane2[neuron] = voltage >= kThreshold ? voltage - kThreshold : voltage;
      voltage_total[neuron] += voltage;
    }
  }

  for (float &voltage : voltage_total) {
    voltage /= kTimesteps;
  }
  float score = dot(kOutWeight[0], voltage_total) + kOutBias[0];

#if SNN_HAS_SKIP
  for (float &feature : token_total) {
    feature /= kTimesteps;
  }
  score += dot(kSkipWeight[0], token_total);
#endif

  return score;
}

}  // namespace

void setup() {
  Serial.setRxBufferSize(sizeof(tokens));
  Serial.begin(500000);
}

void loop() {
  // Drain the receive buffer as bytes arrive. The full input need not fit in it.
  auto *bytes = reinterpret_cast<unsigned char *>(tokens);
  while (received < sizeof(tokens)) {
    const int byte = Serial.read();
    if (byte < 0) {
      return;
    }
    bytes[received++] = static_cast<unsigned char>(byte);
  }

  const int64_t start = esp_timer_get_time();
  const float score = infer();
  const float prediction = snn_model::kTargetMean + 10.0f * score;
  const auto elapsed = static_cast<unsigned long>(esp_timer_get_time() - start);
  received = 0;

  Serial.print("SNN_RESULT ");
  Serial.print(score, 7);
  Serial.print(' ');
  Serial.print(prediction, 7);
  Serial.print(' ');
  Serial.println(elapsed);
  Serial.write(kAck);
}
