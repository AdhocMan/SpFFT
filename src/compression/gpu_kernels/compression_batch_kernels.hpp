#ifndef SPFFT_COMPRESSION_BATCH_KERNELS_HPP
#define SPFFT_COMPRESSION_BATCH_KERNELS_HPP
#include "gpu_util/gpu_fft_api.hpp"
#include "gpu_util/gpu_runtime_api.hpp"
#include "memory/gpu_array_view.hpp"

namespace spfft {

auto compress_batch_gpu(const gpu::StreamType stream, const GPUArrayView1D<int>& indices,
                        const GPUArrayView2D<typename gpu::fft::ComplexType<double>::type>& input,
                        double* output, int batchSize, int singleBatchStickSize, bool useScaling,
                        double scalingFactor) -> void;

auto compress_batch_gpu(const gpu::StreamType stream, const GPUArrayView1D<int>& indices,
                        const GPUArrayView2D<typename gpu::fft::ComplexType<float>::type>& input,
                        float* output, int batchSize, int singleBatchStickSize, bool useScaling,
                        float scalingFactor) -> void;

auto decompress_batch_gpu(const gpu::StreamType stream, const GPUArrayView1D<int>& indices,
                          const double* input,
                          GPUArrayView2D<typename gpu::fft::ComplexType<double>::type> output,
                          int batchSize, int singleBatchStickSize) -> void;

auto decompress_batch_gpu(const gpu::StreamType stream, const GPUArrayView1D<int>& indices,
                          const float* input,
                          GPUArrayView2D<typename gpu::fft::ComplexType<float>::type> output,
                          int batchSize, int singleBatchStickSize) -> void;

}  // namespace spfft

#endif
