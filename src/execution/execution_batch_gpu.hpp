#ifndef SPFFT_EXECUTION_BATCH_GPU_HPP
#define SPFFT_EXECUTION_BATCH_GPU_HPP

#include <complex>
#include <memory>
#include "compression/compression_batch_gpu.hpp"
#include "fft/transform_interface.hpp"
#include "gpu_util/gpu_event_handle.hpp"
#include "gpu_util/gpu_fft_api.hpp"
#include "gpu_util/gpu_runtime_api.hpp"
#include "gpu_util/gpu_stream_handle.hpp"
#include "memory/gpu_array.hpp"
#include "memory/gpu_array_view.hpp"
#include "memory/host_array.hpp"
#include "memory/host_array_view.hpp"
#include "parameters/parameters.hpp"
#include "spfft/config.h"
#include "spfft/types.h"
#include "symmetry/symmetry.hpp"
#include "transpose/transpose.hpp"
#include "util/common_types.hpp"

namespace spfft {

template <typename T>
class ExecutionBatchGPU {
public:
  ExecutionBatchGPU(const int numThreads, SizeType batchSize, std::shared_ptr<Parameters> param,
                    HostArray<std::complex<T>>& array1, HostArray<std::complex<T>>& array2,
                    GPUArray<typename gpu::fft::ComplexType<T>::type>& gpuArray1,
                    GPUArray<typename gpu::fft::ComplexType<T>::type>& gpuArray2,
                    const std::shared_ptr<GPUArray<char>>& fftWorkBuffer);

  auto forward_z(T* output, const SpfftScalingType scalingType) -> void;
  auto forward_exchange(const bool nonBlockingExchange) -> void;
  auto forward_xy(const T* input) -> void;

  auto backward_z(const T* input) -> void;
  auto backward_exchange(const bool nonBlockingExchange) -> void;
  auto backward_xy(T* output) -> void;

  auto synchronize(SpfftExecType mode) -> void;

  auto space_domain_data_host() -> HostArrayView3D<T>;
  auto space_domain_data_gpu() -> GPUArrayView3D<T>;

private:
  GPUStreamHandle stream_;
  gpu::StreamType externalStream_;
  GPUEventHandle startEvent_;
  GPUEventHandle endEvent_;
  int numThreads_;
  SizeType batchSize_;
  T scalingFactor_;

  std::unique_ptr<TransformGPU> transformZ_;
  std::unique_ptr<Transpose> transpose_;
  std::unique_ptr<TransformGPU> transformXY_;

  std::unique_ptr<Symmetry> zStickSymmetry_;
  std::unique_ptr<Symmetry> planeSymmetry_;

  std::unique_ptr<CompressionBatchGPU> compression_;

  HostArrayView3D<T> spaceDomainDataExternalHost_;
  GPUArrayView3D<T> spaceDomainDataExternalGPU_;

  GPUArrayView2D<typename gpu::fft::ComplexType<T>::type> freqDomainDataGPU_;
  GPUArrayView1D<T> freqDomainCompressedDataGPU_;
  GPUArrayView3D<typename gpu::fft::ComplexType<T>::type> freqDomainXYGPU_;
};
}  // namespace spfft
#endif
