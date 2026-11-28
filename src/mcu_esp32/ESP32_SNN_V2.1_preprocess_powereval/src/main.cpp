#include <Arduino.h>
#include <esp_heap_caps.h>
#include <esp_timer.h>
#include <esp32-hal-cpu.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <cstring>
#include <limits>

#include "preprocess.h"
#include "snn_model_data.h"

namespace {
constexpr size_t kTimesteps = preprocess::kSteps;
constexpr size_t kFeatures = preprocess::kFeatures;
constexpr float kLeak = 0.875f;
constexpr float kThreshold = 0.5f;
constexpr uint8_t kAck = 0xaa;
constexpr unsigned long kSettleMs = 2000;
constexpr int64_t kYieldEveryUs = 100000;
constexpr size_t kChunkBytes = 512;
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
              "The token CRC uses IEEE-754 binary32");
static_assert(preprocess::kFrameBytes % kChunkBytes == 0, "Require full chunks");
float tokens[kTimesteps][kFeatures];
int16_t *events[preprocess::kEvents]{};
const int16_t *event_views[preprocess::kEvents]{};
bool allocated = false;

bool allocate_frame() {
  if (allocated) return true;
  for (size_t e = 0; e < preprocess::kEvents; ++e) {
    events[e] = static_cast<int16_t *>(
        heap_caps_malloc(preprocess::kEventBytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    if (events[e] == nullptr) {
      for (size_t i = 0; i < e; ++i) {
        heap_caps_free(events[i]);
        events[i] = nullptr;
        event_views[i] = nullptr;
      }
      return false;
    }
    event_views[e] = events[e];
  }
  allocated = true;
  return true;
}

uint16_t crc_byte(uint16_t crc, uint8_t byte) {
  crc ^= static_cast<uint16_t>(byte) << 8;
  for (int bit = 0; bit < 8; ++bit) {
    crc = static_cast<uint16_t>((crc << 1) ^ ((crc & 0x8000) ? 0x1021 : 0));
  }
  return crc;
}

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

void power_benchmark(uint32_t duration_ms) {
  Serial.print("SNN_POWER_START ");
  Serial.print(static_cast<unsigned long>(kSettleMs));
  Serial.print(' ');
  Serial.print(static_cast<unsigned long>(duration_ms));
  Serial.print(' ');
  Serial.println(static_cast<unsigned long>(getCpuFrequencyMhz()));
  Serial.flush();
  delay(kSettleMs);

  const int64_t start = esp_timer_get_time();
  int64_t now = start;
  int64_t last_yield = start;
  const int64_t deadline = start + static_cast<int64_t>(duration_ms) * 1000;
  unsigned long count = 0;
  float score = 0.0f;
  do {
    preprocess::encode(event_views, tokens);
    score = infer();
    // Keep each call observable and prevent hoisting it out of the loop.
    asm volatile("" : : "m"(score) : "memory");
    ++count;
    now = esp_timer_get_time();
    if (now < deadline && now - last_yield >= kYieldEveryUs) {
      vTaskDelay(1);  // Give the idle task time to run.
      now = esp_timer_get_time();
      last_yield = now;
    }
  } while (now < deadline);

  Serial.print("SNN_POWER_RESULT ");
  Serial.print(count);
  Serial.print(' ');
  Serial.print(static_cast<unsigned long>(now - start));
  Serial.print(' ');
  Serial.print(score, 7);
  Serial.print(' ');
  // Verify the ESP32's pooled, pre-gain tokens without counting CRC in the timing.
  const auto *token_bytes = reinterpret_cast<const uint8_t *>(tokens);
  uint16_t token_crc = 0xffff;
  for (size_t i = 0; i < sizeof(tokens); ++i) {
    token_crc = crc_byte(token_crc, token_bytes[i]);
  }
  Serial.println(static_cast<unsigned long>(token_crc));
  Serial.write(kAck);
}

}  // namespace

void setup() {
  Serial.setRxBufferSize(4096);
  Serial.begin(500000);
}

void loop() {
  static uint8_t header[8];
  static size_t header_bytes = 0;
  static size_t received = 0;
  static uint32_t duration_ms = 0;
  static uint16_t input_crc = 0xffff;
  while (true) {
    const int byte = Serial.read();
    if (byte < 0) return;
    if (duration_ms == 0) {
      header[header_bytes++] = byte;
      if (header_bytes != sizeof(header)) continue;
      header_bytes = 0;
      if (std::memcmp(header, "RF21", 4) != 0) {
        Serial.println("SNN_POWER_ERROR expected RF21 benchmark command; reset before retrying");
        return;
      }
      const uint32_t requested = uint32_t{header[4]} | (uint32_t{header[5]} << 8)
                               | (uint32_t{header[6]} << 16) | (uint32_t{header[7]} << 24);
      if (requested < 1000 || requested > 60000) {
        Serial.println("SNN_POWER_ERROR duration must be 1000 to 60000 ms");
        return;
      }
      if (!allocate_frame()) {
        Serial.print("SNN_POWER_ERROR insufficient internal RAM for 250 KiB of RF samples; free bytes ");
        Serial.println(static_cast<unsigned long>(
            heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)));
        return;
      }
      duration_ms = requested;
      received = 0;
      input_crc = 0xffff;
      Serial.println("SNN_RF_READY");
      continue;
    }
    auto *event = reinterpret_cast<uint8_t *>(events[received / preprocess::kEventBytes]);
    event[received % preprocess::kEventBytes] = byte;
    input_crc = crc_byte(input_crc, byte);
    ++received;
    if (received == preprocess::kFrameBytes) {
      Serial.print("SNN_RF_CRC ");
      Serial.println(static_cast<unsigned long>(input_crc));
      const uint32_t window = duration_ms;
      duration_ms = 0;
      power_benchmark(window);
      return;
    }
    if (received % kChunkBytes == 0) {
      Serial.print("SNN_RF_RX ");
      Serial.println(static_cast<unsigned long>(received));
    }
  }
}
