#ifndef SPFFT_COMPRESSION_BATCH_HOST_HPP
#define SPFFT_COMPRESSION_BATCH_HOST_HPP

#include <complex>
#include <cstring>
#include <memory>
#include "memory/host_array_view.hpp"
#include "parameters/parameters.hpp"
#include "spfft/config.h"
#include "util/common_types.hpp"
#include "util/omp_definitions.hpp"

namespace spfft {

class CompressionBatchHost {
public:
  CompressionBatchHost(const std::shared_ptr<Parameters>& param, SizeType batchSize)
      : param_(param), batchSize_(batchSize) {}

  template <typename T>
  auto compress(const HostArrayView2D<std::complex<T>> input2d, T* output, bool useScaling,
                const T scalingFactor = 1.0) const -> void {
    const auto& indices = param_->local_value_indices();
    const SizeType numElements = indices.size();
    const SizeType singleBatchStickSize = input2d.dim_outer() / batchSize_ * input2d.dim_inner();

    if (useScaling) {
      SPFFT_OMP_PRAGMA("omp for schedule(static)")
      for (SizeType bi = 0; bi < batchSize_ * numElements; ++bi) {
        const SizeType b = bi / numElements;
        const SizeType i = bi % numElements;
        const auto value =
            scalingFactor *
            input2d.data()[b * singleBatchStickSize + indices[i]];
        output[b * 2 * numElements + 2 * i] = value.real();
        output[b * 2 * numElements + 2 * i + 1] = value.imag();
      }
    } else {
      SPFFT_OMP_PRAGMA("omp for schedule(static)")
      for (SizeType bi = 0; bi < batchSize_ * numElements; ++bi) {
        const SizeType b = bi / numElements;
        const SizeType i = bi % numElements;
        const auto value =
            input2d.data()[b * singleBatchStickSize + indices[i]];
        output[b * 2 * numElements + 2 * i] = value.real();
        output[b * 2 * numElements + 2 * i + 1] = value.imag();
      }
    }
  }

  template <typename T>
  auto decompress(const T* input, HostArrayView2D<std::complex<T>> output2d) const -> void {
    const auto& indices = param_->local_value_indices();
    const SizeType numElements = indices.size();
    const SizeType numSticksPerBatch = output2d.dim_outer() / batchSize_;
    const SizeType singleBatchStickSize = numSticksPerBatch * output2d.dim_inner();

    SPFFT_OMP_PRAGMA("omp for schedule(static)")
    for (SizeType stick = 0; stick < output2d.dim_outer(); ++stick) {
      std::memset(static_cast<void*>(&output2d(stick, 0)), 0,
                  sizeof(typename decltype(output2d)::ValueType) * output2d.dim_inner());
    }

    SPFFT_OMP_PRAGMA("omp for schedule(static)")
    for (SizeType bi = 0; bi < batchSize_ * numElements; ++bi) {
      const SizeType b = bi / numElements;
      const SizeType i = bi % numElements;
      output2d.data()[b * singleBatchStickSize + indices[i]] =
          std::complex<T>(input[b * 2 * numElements + 2 * i],
                          input[b * 2 * numElements + 2 * i + 1]);
    }
  }

private:
  std::shared_ptr<Parameters> param_;
  SizeType batchSize_;
};
}  // namespace spfft

#endif
