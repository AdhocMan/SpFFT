#ifndef SPFFT_COMPRESSION_BATCH_GPU_HPP
#define SPFFT_COMPRESSION_BATCH_GPU_HPP

#include <memory>
#include "compression/gpu_kernels/compression_batch_kernels.hpp"
#include "gpu_util/gpu_fft_api.hpp"
#include "gpu_util/gpu_stream_handle.hpp"
#include "gpu_util/gpu_transfer.hpp"
#include "memory/array_view_utility.hpp"
#include "memory/gpu_array.hpp"
#include "memory/gpu_array_view.hpp"
#include "parameters/parameters.hpp"
#include "spfft/config.h"
#include "util/common_types.hpp"
#include "util/type_check.hpp"

namespace spfft {

class CompressionBatchGPU {
public:
  CompressionBatchGPU(const std::shared_ptr<Parameters>& param, SizeType batchSize)
      : indicesGPU_(param->local_value_indices().size()),
        batchSize_(batchSize),
        numZSticks_(param->num_z_sticks(0)),
        dimZ_(param->dim_z()) {
    copy_to_gpu(param->local_value_indices(), indicesGPU_);
  }

  template <typename T>
  auto compress(const GPUStreamHandle& stream,
                const GPUArrayView2D<typename gpu::fft::ComplexType<T>::type> input, T* output,
                const bool useScaling, const T scalingFactor = 1.0) -> void {
    static_assert(IsFloatOrDouble<T>::value, "Type T must be float or double");
    const int singleBatchStickSize = numZSticks_ * dimZ_;
    compress_batch_gpu(stream.get(), create_1d_view(indicesGPU_, 0, indicesGPU_.size()), input,
                       output, static_cast<int>(batchSize_), singleBatchStickSize, useScaling,
                       scalingFactor);
  }

  template <typename T>
  auto decompress(const GPUStreamHandle& stream, const T* input,
                  GPUArrayView2D<typename gpu::fft::ComplexType<T>::type> output) -> void {
    static_assert(IsFloatOrDouble<T>::value, "Type T must be float or double");
    gpu::check_status(gpu::memset_async(
        static_cast<void*>(output.data()), 0,
        output.size() * sizeof(typename decltype(output)::ValueType), stream.get()));
    const int singleBatchStickSize = numZSticks_ * dimZ_;
    decompress_batch_gpu(stream.get(), create_1d_view(indicesGPU_, 0, indicesGPU_.size()), input,
                         output, static_cast<int>(batchSize_), singleBatchStickSize);
  }

private:
  GPUArray<int> indicesGPU_;
  SizeType batchSize_;
  SizeType numZSticks_;
  SizeType dimZ_;
};
}  // namespace spfft

#endif
