/*
 * Copyright (c) 2019 ETH Zurich, Simon Frasch
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
#include <algorithm>
#include <cassert>
#include "gpu_util/gpu_fft_api.hpp"
#include "gpu_util/gpu_kernel_parameter.hpp"
#include "gpu_util/gpu_runtime.hpp"
#include "memory/array_view_utility.hpp"
#include "memory/gpu_array_const_view.hpp"
#include "spfft/config.h"

namespace spfft {

// ------------------
// Backward
// ------------------

#ifdef SPFFT_CUDA
// kernel optimized for NVIDIA
template <typename T>
__global__ static void transpose_batch_backward_kernel(
    const GPUArrayConstView1D<int> indices, const T* freqZData, T* spaceDomainFlat,
    const int numZSticks, const int dimZ, const int xyPlaneSize, const int batchSize) {
  const int bStickTotal = batchSize * numZSticks;
  const int stickIndex = threadIdx.x + blockIdx.x * blockDim.x;

  if (stickIndex < bStickTotal) {
    const int b = stickIndex / numZSticks;
    const int s = stickIndex - b * numZSticks;
    const auto stickXYIndex = indices(s);
    for (int z = blockIdx.y; z < dimZ; z += gridDim.y) {
      spaceDomainFlat[b * dimZ * xyPlaneSize + z * xyPlaneSize + stickXYIndex] =
          freqZData[b * numZSticks * dimZ + s * dimZ + z];
    }
  }
}

#else
// kernel optimized for AMD
template <typename T>
__global__ static void transpose_batch_backward_kernel(
    const GPUArrayConstView1D<int> indices, const T* freqZData, T* spaceDomainFlat,
    const int numZSticks, const int dimZ, const int xyPlaneSize, const int batchSize) {
  const int bStickTotal = batchSize * numZSticks;
  const int z = threadIdx.x + blockIdx.x * blockDim.x;

  if (z < dimZ) {
    for (int bStick = blockIdx.y; bStick < bStickTotal; bStick += gridDim.y) {
      const int b = bStick / numZSticks;
      const int s = bStick - b * numZSticks;
      const auto stickXYIndex = indices(s);
      spaceDomainFlat[b * dimZ * xyPlaneSize + z * xyPlaneSize + stickXYIndex] =
          freqZData[b * numZSticks * dimZ + s * dimZ + z];
    }
  }
}
#endif

template <typename T>
static void local_transpose_batch_backward_impl(
    const gpu::StreamType stream, const GPUArrayView1D<int>& indices, const T* freqZData,
    T* spaceDomainFlat, const int numZSticks, const int dimZ, const int xyPlaneSize,
    const int batchSize) {
  const int bStickTotal = batchSize * numZSticks;
  const dim3 threadBlock(gpu::BlockSizeSmall);
#ifdef SPFFT_CUDA
  const dim3 threadGrid((bStickTotal + threadBlock.x - 1) / threadBlock.x,
                        std::min(dimZ, gpu::GridSizeMedium));
#else
  const dim3 threadGrid((dimZ + threadBlock.x - 1) / threadBlock.x,
                        std::min(bStickTotal, gpu::GridSizeMedium));
#endif
  launch_kernel(transpose_batch_backward_kernel<T>, threadGrid, threadBlock, 0, stream,
                GPUArrayConstView1D<int>(indices), freqZData, spaceDomainFlat, numZSticks, dimZ,
                xyPlaneSize, batchSize);
}

// ------------------
// Forward
// ------------------

template <typename T>
__global__ static void transpose_batch_forward_kernel(
    const GPUArrayConstView1D<int> indices, const T* spaceDomainFlat, T* freqZData,
    const int numZSticks, const int dimZ, const int xyPlaneSize, const int batchSize) {
  const int bStickTotal = batchSize * numZSticks;
  const int z = threadIdx.x + blockIdx.x * blockDim.x;

  if (z < dimZ) {
    for (int bStick = blockIdx.y; bStick < bStickTotal; bStick += gridDim.y) {
      const int b = bStick / numZSticks;
      const int s = bStick - b * numZSticks;
      const auto stickXYIndex = indices(s);
      freqZData[b * numZSticks * dimZ + s * dimZ + z] =
          spaceDomainFlat[b * dimZ * xyPlaneSize + z * xyPlaneSize + stickXYIndex];
    }
  }
}

template <typename T>
static void local_transpose_batch_forward_impl(
    const gpu::StreamType stream, const GPUArrayView1D<int>& indices, const T* spaceDomainFlat,
    T* freqZData, const int numZSticks, const int dimZ, const int xyPlaneSize,
    const int batchSize) {
  const int bStickTotal = batchSize * numZSticks;
  const dim3 threadBlock(gpu::BlockSizeSmall);
  const dim3 threadGrid((dimZ + threadBlock.x - 1) / threadBlock.x,
                        std::min(bStickTotal, gpu::GridSizeMedium));
  launch_kernel(transpose_batch_forward_kernel<T>, threadGrid, threadBlock, 0, stream,
                GPUArrayConstView1D<int>(indices), spaceDomainFlat, freqZData, numZSticks, dimZ,
                xyPlaneSize, batchSize);
}

// ------------------
// Public overloads
// ------------------

auto local_transpose_batch_backward(
    const gpu::StreamType stream, const GPUArrayView1D<int>& indices,
    const typename gpu::fft::ComplexType<double>::type* freqZData,
    typename gpu::fft::ComplexType<double>::type* spaceDomainFlat, const int numZSticks,
    const int dimZ, const int xyPlaneSize, const int batchSize) -> void {
  local_transpose_batch_backward_impl(stream, indices, freqZData, spaceDomainFlat, numZSticks,
                                      dimZ, xyPlaneSize, batchSize);
}

auto local_transpose_batch_backward(
    const gpu::StreamType stream, const GPUArrayView1D<int>& indices,
    const typename gpu::fft::ComplexType<float>::type* freqZData,
    typename gpu::fft::ComplexType<float>::type* spaceDomainFlat, const int numZSticks,
    const int dimZ, const int xyPlaneSize, const int batchSize) -> void {
  local_transpose_batch_backward_impl(stream, indices, freqZData, spaceDomainFlat, numZSticks,
                                      dimZ, xyPlaneSize, batchSize);
}

auto local_transpose_batch_forward(
    const gpu::StreamType stream, const GPUArrayView1D<int>& indices,
    const typename gpu::fft::ComplexType<double>::type* spaceDomainFlat,
    typename gpu::fft::ComplexType<double>::type* freqZData, const int numZSticks, const int dimZ,
    const int xyPlaneSize, const int batchSize) -> void {
  local_transpose_batch_forward_impl(stream, indices, spaceDomainFlat, freqZData, numZSticks, dimZ,
                                     xyPlaneSize, batchSize);
}

auto local_transpose_batch_forward(
    const gpu::StreamType stream, const GPUArrayView1D<int>& indices,
    const typename gpu::fft::ComplexType<float>::type* spaceDomainFlat,
    typename gpu::fft::ComplexType<float>::type* freqZData, const int numZSticks, const int dimZ,
    const int xyPlaneSize, const int batchSize) -> void {
  local_transpose_batch_forward_impl(stream, indices, spaceDomainFlat, freqZData, numZSticks, dimZ,
                                     xyPlaneSize, batchSize);
}

}  // namespace spfft
