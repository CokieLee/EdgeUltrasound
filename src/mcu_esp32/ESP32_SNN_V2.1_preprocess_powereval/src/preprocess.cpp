#include "preprocess.h"

namespace preprocess {
void encode(const int16_t *const (&events)[kEvents],
            float (&tokens)[kSteps][kFeatures]) {
  constexpr std::size_t kGroupChannels = 8;
  constexpr std::size_t kBinSamples = 16;
  constexpr std::size_t kGroups = kChannels / kGroupChannels;
  constexpr float kInverseBinSize = 1.0f / (kGroupChannels * kBinSamples);

  for (std::size_t e = 0; e < kEvents; ++e) {
    for (std::size_t group = 0; group < kGroups; ++group) {
      uint32_t sums[kSteps]{};
      for (std::size_t c = 0; c < kGroupChannels; ++c) {
        const auto *channel = events[e] + (group * kGroupChannels + c) * kSamples;
        for (std::size_t i = 0; i < kSamples; ++i) {
          const int32_t sample = channel[i];
          sums[i / kBinSamples] += sample < 0 ? -sample : sample;
        }
      }
      // The final bin contains ten samples and six implicit zeros per channel.
      for (std::size_t t = 0; t < kSteps; ++t) {
        tokens[t][e * kGroups + group] = sums[t] * kInverseBinSize;
      }
    }
  }
}
}  // namespace preprocess
