#ifndef SPFFT_SYMMETRY_BATCH_HOST_HPP
#define SPFFT_SYMMETRY_BATCH_HOST_HPP

#include <complex>
#include "memory/host_array_view.hpp"
#include "spfft/config.h"
#include "symmetry/symmetry.hpp"
#include "util/common_types.hpp"
#include "util/omp_definitions.hpp"

namespace spfft {

template <typename T>
class StickSymmetryBatchHost : public Symmetry {
public:
  StickSymmetryBatchHost(HostArrayView2D<std::complex<T>> freqDomainData,
                         SizeType zeroZeroStickIndex, SizeType numZSticks, SizeType batchSize)
      : freqDomainData_(freqDomainData),
        zeroZeroStickIndex_(zeroZeroStickIndex),
        numZSticks_(numZSticks),
        batchSize_(batchSize) {}

  auto apply() -> void override {
    constexpr std::complex<T> zeroElement;
    const SizeType stickLen = freqDomainData_.dim_inner();

    SPFFT_OMP_PRAGMA("omp for schedule(static)")
    for (SizeType b = 0; b < batchSize_; ++b) {
      const SizeType row = b * numZSticks_ + zeroZeroStickIndex_;
      for (SizeType idx = 1; idx < stickLen / 2 + 1; ++idx) {
        const auto value = freqDomainData_(row, idx);
        if (value != zeroElement) {
          freqDomainData_(row, stickLen - idx) = std::conj(value);
        }
      }
      for (SizeType idx = stickLen / 2 + 1; idx < stickLen; ++idx) {
        const auto value = freqDomainData_(row, idx);
        if (value != zeroElement) {
          freqDomainData_(row, stickLen - idx) = std::conj(value);
        }
      }
    }
  }

private:
  HostArrayView2D<std::complex<T>> freqDomainData_;
  SizeType zeroZeroStickIndex_;
  SizeType numZSticks_;
  SizeType batchSize_;
};
}  // namespace spfft

#endif
