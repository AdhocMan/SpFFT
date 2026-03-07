#ifndef SPFFT_SYMMETRY_BATCH_KERNELS_HPP
#define SPFFT_SYMMETRY_BATCH_KERNELS_HPP
#include "gpu_util/gpu_fft_api.hpp"
#include "gpu_util/gpu_runtime_api.hpp"
#include "memory/gpu_array_view.hpp"

namespace spfft {

auto symmetrize_stick_batch_gpu(
    const gpu::StreamType stream,
    GPUArrayView2D<typename gpu::fft::ComplexType<double>::type> freqDomainData,
    int zeroZeroStickIndex, int numZSticks, int batchSize) -> void;

auto symmetrize_stick_batch_gpu(
    const gpu::StreamType stream,
    GPUArrayView2D<typename gpu::fft::ComplexType<float>::type> freqDomainData,
    int zeroZeroStickIndex, int numZSticks, int batchSize) -> void;

}  // namespace spfft

#endif
