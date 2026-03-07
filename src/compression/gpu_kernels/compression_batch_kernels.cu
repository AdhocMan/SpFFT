#include <algorithm>
#include <cassert>

#include "gpu_util/gpu_fft_api.hpp"
#include "gpu_util/gpu_kernel_parameter.hpp"
#include "gpu_util/gpu_runtime.hpp"
#include "memory/gpu_array_const_view.hpp"
#include "memory/gpu_array_view.hpp"

namespace spfft {

template <typename T>
__global__ static void decompress_batch_kernel(
    const GPUArrayConstView1D<int> indices, const T* input,
    GPUArrayView1D<typename gpu::fft::ComplexType<T>::type> output, const int batchSize,
    const int singleBatchStickSize) {
  const int numElements = indices.size();
  const int totalWork = batchSize * numElements;
  for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < totalWork;
       idx += gridDim.x * blockDim.x) {
    const int b = idx / numElements;
    const int i = idx % numElements;
    const int valueIdx = b * singleBatchStickSize + indices(i);
    typename gpu::fft::ComplexType<T>::type value;
    value.x = input[b * 2 * numElements + 2 * i];
    value.y = input[b * 2 * numElements + 2 * i + 1];
    output(valueIdx) = value;
  }
}

template <typename T>
__global__ static void compress_batch_kernel(
    const GPUArrayConstView1D<int> indices,
    GPUArrayConstView1D<typename gpu::fft::ComplexType<T>::type> input, T* output,
    const int batchSize, const int singleBatchStickSize) {
  const int numElements = indices.size();
  const int totalWork = batchSize * numElements;
  for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < totalWork;
       idx += gridDim.x * blockDim.x) {
    const int b = idx / numElements;
    const int i = idx % numElements;
    const int valueIdx = b * singleBatchStickSize + indices(i);
    const auto value = input(valueIdx);
    output[b * 2 * numElements + 2 * i] = value.x;
    output[b * 2 * numElements + 2 * i + 1] = value.y;
  }
}

template <typename T>
__global__ static void compress_batch_kernel_scaled(
    const GPUArrayConstView1D<int> indices,
    GPUArrayConstView1D<typename gpu::fft::ComplexType<T>::type> input, T* output,
    const int batchSize, const int singleBatchStickSize, const T scalingFactor) {
  const int numElements = indices.size();
  const int totalWork = batchSize * numElements;
  for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < totalWork;
       idx += gridDim.x * blockDim.x) {
    const int b = idx / numElements;
    const int i = idx % numElements;
    const int valueIdx = b * singleBatchStickSize + indices(i);
    const auto value = input(valueIdx);
    output[b * 2 * numElements + 2 * i] = scalingFactor * value.x;
    output[b * 2 * numElements + 2 * i + 1] = scalingFactor * value.y;
  }
}

template <typename T>
static void compress_batch_gpu_impl(const gpu::StreamType stream,
                                    const GPUArrayView1D<int>& indices,
                                    const GPUArrayView2D<typename gpu::fft::ComplexType<T>::type>& input,
                                    T* output, int batchSize, int singleBatchStickSize,
                                    bool useScaling, T scalingFactor) {
  const int totalWork = batchSize * indices.size();
  const dim3 threadBlock(gpu::BlockSizeMedium);
  const dim3 threadGrid(
      std::min(static_cast<int>((totalWork + threadBlock.x - 1) / threadBlock.x),
               gpu::GridSizeMedium));

  if (useScaling) {
    launch_kernel(compress_batch_kernel_scaled<T>, threadGrid, threadBlock, 0, stream,
                  GPUArrayConstView1D<int>(indices),
                  GPUArrayConstView1D<typename gpu::fft::ComplexType<T>::type>(
                      input.data(), input.size(), input.device_id()),
                  output, batchSize, singleBatchStickSize, scalingFactor);
  } else {
    launch_kernel(compress_batch_kernel<T>, threadGrid, threadBlock, 0, stream,
                  GPUArrayConstView1D<int>(indices),
                  GPUArrayConstView1D<typename gpu::fft::ComplexType<T>::type>(
                      input.data(), input.size(), input.device_id()),
                  output, batchSize, singleBatchStickSize);
  }
}

template <typename T>
static void decompress_batch_gpu_impl(const gpu::StreamType stream,
                                      const GPUArrayView1D<int>& indices, const T* input,
                                      GPUArrayView2D<typename gpu::fft::ComplexType<T>::type> output,
                                      int batchSize, int singleBatchStickSize) {
  const int totalWork = batchSize * indices.size();
  const dim3 threadBlock(gpu::BlockSizeMedium);
  const dim3 threadGrid(
      std::min(static_cast<int>((totalWork + threadBlock.x - 1) / threadBlock.x),
               gpu::GridSizeMedium));

  launch_kernel(decompress_batch_kernel<T>, threadGrid, threadBlock, 0, stream,
                GPUArrayConstView1D<int>(indices), input,
                GPUArrayView1D<typename gpu::fft::ComplexType<T>::type>(
                    output.data(), output.size(), output.device_id()),
                batchSize, singleBatchStickSize);
}

auto compress_batch_gpu(const gpu::StreamType stream, const GPUArrayView1D<int>& indices,
                        const GPUArrayView2D<typename gpu::fft::ComplexType<double>::type>& input,
                        double* output, int batchSize, int singleBatchStickSize, bool useScaling,
                        double scalingFactor) -> void {
  compress_batch_gpu_impl<double>(stream, indices, input, output, batchSize, singleBatchStickSize,
                                  useScaling, scalingFactor);
}

auto compress_batch_gpu(const gpu::StreamType stream, const GPUArrayView1D<int>& indices,
                        const GPUArrayView2D<typename gpu::fft::ComplexType<float>::type>& input,
                        float* output, int batchSize, int singleBatchStickSize, bool useScaling,
                        float scalingFactor) -> void {
  compress_batch_gpu_impl<float>(stream, indices, input, output, batchSize, singleBatchStickSize,
                                 useScaling, scalingFactor);
}

auto decompress_batch_gpu(const gpu::StreamType stream, const GPUArrayView1D<int>& indices,
                          const double* input,
                          GPUArrayView2D<typename gpu::fft::ComplexType<double>::type> output,
                          int batchSize, int singleBatchStickSize) -> void {
  decompress_batch_gpu_impl<double>(stream, indices, input, output, batchSize,
                                    singleBatchStickSize);
}

auto decompress_batch_gpu(const gpu::StreamType stream, const GPUArrayView1D<int>& indices,
                          const float* input,
                          GPUArrayView2D<typename gpu::fft::ComplexType<float>::type> output,
                          int batchSize, int singleBatchStickSize) -> void {
  decompress_batch_gpu_impl<float>(stream, indices, input, output, batchSize,
                                   singleBatchStickSize);
}

}  // namespace spfft
