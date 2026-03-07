#ifndef SPFFT_BATCH_TRANSFORM_INTERNAL_HPP
#define SPFFT_BATCH_TRANSFORM_INTERNAL_HPP

#include <memory>
#include "execution/execution_batch_host.hpp"
#include "memory/host_array.hpp"
#include "parameters/parameters.hpp"
#include "spfft/config.h"
#include "spfft/types.h"
#include "util/common_types.hpp"

#if defined(SPFFT_CUDA) || defined(SPFFT_ROCM)
#include "execution/execution_batch_gpu.hpp"
#include "memory/gpu_array.hpp"
#include "gpu_util/gpu_fft_api.hpp"
#endif

namespace spfft {
template <typename T>
class BatchTransformInternal {
public:
  BatchTransformInternal(int maxNumThreads, SpfftProcessingUnitType processingUnit,
                         SpfftTransformType transformType, int dimX, int dimY, int dimZ,
                         int batchSize, int numLocalElements, SpfftIndexFormatType indexFormat,
                         const int* indices);

  auto forward(const T* input, T* output, SpfftScalingType scaling) -> void;
  auto forward(SpfftProcessingUnitType inputLocation, T* output, SpfftScalingType scaling) -> void;

  auto backward(const T* input, T* output) -> void;
  auto backward(const T* input, SpfftProcessingUnitType outputLocation) -> void;

  auto space_domain_data(SpfftProcessingUnitType pu) -> T*;

  inline auto batch_size() const noexcept -> int { return batchSize_; }
  inline auto dim_x() const noexcept -> int { return param_->dim_x(); }
  inline auto dim_y() const noexcept -> int { return param_->dim_y(); }
  inline auto dim_z() const noexcept -> int { return param_->dim_z(); }
  inline auto num_local_elements() const noexcept -> int { return param_->local_num_elements(); }
  inline auto type() const noexcept -> SpfftTransformType { return param_->transform_type(); }
  inline auto num_threads() const noexcept -> int { return numThreads_; }
  inline auto processing_unit() const noexcept -> SpfftProcessingUnitType { return processingUnit_; }

private:
  int batchSize_;
  int numThreads_;
  SpfftProcessingUnitType processingUnit_;
  std::shared_ptr<Parameters> param_;
  HostArray<std::complex<T>> arrayHost1_;
  HostArray<std::complex<T>> arrayHost2_;
  std::unique_ptr<ExecutionBatchHost<T>> execHost_;
#if defined(SPFFT_CUDA) || defined(SPFFT_ROCM)
  GPUArray<typename gpu::fft::ComplexType<T>::type> gpuArray1_;
  GPUArray<typename gpu::fft::ComplexType<T>::type> gpuArray2_;
  std::shared_ptr<GPUArray<char>> fftWorkBuffer_;
  std::unique_ptr<ExecutionBatchGPU<T>> execGPU_;
#endif
};

}  // namespace spfft

#endif
