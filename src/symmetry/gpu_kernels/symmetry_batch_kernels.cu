#include <algorithm>
#include <cassert>
#include "gpu_util/gpu_fft_api.hpp"
#include "gpu_util/gpu_kernel_parameter.hpp"
#include "gpu_util/gpu_runtime.hpp"
#include "memory/gpu_array_view.hpp"

namespace spfft {

template <typename T>
__global__ static void symmetrize_stick_batch_kernel(
    GPUArrayView2D<typename gpu::fft::ComplexType<T>::type> freqDomainData,
    const int zeroZeroStickIndex, const int numZSticks, const int batchSize, const int startIndex,
    const int numIndices) {
  const int stickLen = freqDomainData.dim_inner();
  for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < batchSize * numIndices;
       idx += gridDim.x * blockDim.x) {
    const int b = idx / numIndices;
    const int i = idx % numIndices;
    const int row = b * numZSticks + zeroZeroStickIndex;
    const int srcIdx = startIndex + i;
    auto value = freqDomainData(row, srcIdx);
    if (value.x != T(0) || value.y != T(0)) {
      value.y = -value.y;
      freqDomainData(row, stickLen - srcIdx) = value;
    }
  }
}

template <typename T>
static void symmetrize_stick_batch_gpu_impl(
    const gpu::StreamType stream,
    GPUArrayView2D<typename gpu::fft::ComplexType<T>::type> freqDomainData,
    int zeroZeroStickIndex, int numZSticks, int batchSize) {
  const int stickLen = freqDomainData.dim_inner();
  if (stickLen <= 2) return;

  {
    const int startIndex = 1;
    const int numIndices = stickLen / 2;
    const int totalWork = batchSize * numIndices;
    const dim3 threadBlock(gpu::BlockSizeSmall);
    const dim3 threadGrid(std::min(
        static_cast<int>((totalWork + threadBlock.x - 1) / threadBlock.x), gpu::GridSizeMedium));
    launch_kernel(symmetrize_stick_batch_kernel<T>, threadGrid, threadBlock, 0, stream,
                  freqDomainData, zeroZeroStickIndex, numZSticks, batchSize, startIndex,
                  numIndices);
  }
  {
    const int startIndex = stickLen / 2 + 1;
    const int numIndices = stickLen - startIndex;
    if (numIndices > 0) {
      const int totalWork = batchSize * numIndices;
      const dim3 threadBlock(gpu::BlockSizeSmall);
      const dim3 threadGrid(std::min(
          static_cast<int>((totalWork + threadBlock.x - 1) / threadBlock.x), gpu::GridSizeMedium));
      launch_kernel(symmetrize_stick_batch_kernel<T>, threadGrid, threadBlock, 0, stream,
                    freqDomainData, zeroZeroStickIndex, numZSticks, batchSize, startIndex,
                    numIndices);
    }
  }
}

auto symmetrize_stick_batch_gpu(
    const gpu::StreamType stream,
    GPUArrayView2D<typename gpu::fft::ComplexType<double>::type> freqDomainData,
    int zeroZeroStickIndex, int numZSticks, int batchSize) -> void {
  symmetrize_stick_batch_gpu_impl<double>(stream, freqDomainData, zeroZeroStickIndex, numZSticks,
                                          batchSize);
}

auto symmetrize_stick_batch_gpu(
    const gpu::StreamType stream,
    GPUArrayView2D<typename gpu::fft::ComplexType<float>::type> freqDomainData,
    int zeroZeroStickIndex, int numZSticks, int batchSize) -> void {
  symmetrize_stick_batch_gpu_impl<float>(stream, freqDomainData, zeroZeroStickIndex, numZSticks,
                                         batchSize);
}

}  // namespace spfft
