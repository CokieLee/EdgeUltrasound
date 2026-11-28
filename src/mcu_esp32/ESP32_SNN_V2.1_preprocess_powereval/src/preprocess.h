#pragma once

#include <cstddef>
#include <cstdint>

namespace preprocess {
constexpr std::size_t kEvents = 8;
constexpr std::size_t kChannels = 64;
constexpr std::size_t kSamples = 250;
constexpr std::size_t kSteps = 16;
constexpr std::size_t kFeatures = 64;
constexpr std::size_t kEventBytes = kChannels * kSamples * sizeof(int16_t);
constexpr std::size_t kFrameBytes = kEvents * kEventBytes;

void encode(const int16_t *const (&events)[kEvents],
            float (&tokens)[kSteps][kFeatures]);
}  // namespace preprocess
