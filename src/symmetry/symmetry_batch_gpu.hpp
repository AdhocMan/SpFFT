#ifndef SPFFT_SYMMETRY_BATCH_GPU_HPP
#define SPFFT_SYMMETRY_BATCH_GPU_HPP

#include "gpu_util/gpu_fft_api.hpp"
#include "gpu_util/gpu_stream_handle.hpp"
#include "memory/gpu_array_view.hpp"
#include "spfft/config.h"
#include "symmetry/gpu_kernels/symmetry_batch_kernels.hpp"
#include "symmetry/symmetry.hpp"
#include "util/common_types.hpp"

namespace spfft {

template <typename T>
class StickSymmetryBatchGPU : public Symmetry {
public:
  StickSymmetryBatchGPU(
      GPUStreamHandle stream,
      GPUArrayView2D<typename gpu::fft::ComplexType<T>::type> freqDomainData,
      SizeType zeroZeroStickIndex, SizeType numZSticks, SizeType batchSize)
      : stream_(std::move(stream)),
        freqDomainData_(freqDomainData),
        zeroZeroStickIndex_(zeroZeroStickIndex),
        numZSticks_(numZSticks),
        batchSize_(batchSize) {}

  auto apply() -> void override {
    symmetrize_stick_batch_gpu(stream_.get(), freqDomainData_,
                               static_cast<int>(zeroZeroStickIndex_),
                               static_cast<int>(numZSticks_),
                               static_cast<int>(batchSize_));
  }

private:
  GPUStreamHandle stream_;
  GPUArrayView2D<typename gpu::fft::ComplexType<T>::type> freqDomainData_;
  SizeType zeroZeroStickIndex_;
  SizeType numZSticks_;
  SizeType batchSize_;
};
}  // namespace spfft

#endif
