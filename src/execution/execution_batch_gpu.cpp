#include "execution/execution_batch_gpu.hpp"
#include "fft/transform_1d_gpu.hpp"
#include "fft/transform_2d_gpu.hpp"
#include "fft/transform_real_2d_gpu.hpp"
#include "gpu_util/gpu_pointer_translation.hpp"
#include "gpu_util/gpu_runtime_api.hpp"
#include "gpu_util/gpu_transfer.hpp"
#include "memory/array_view_utility.hpp"
#include "parameters/parameters.hpp"
#include "spfft/exceptions.hpp"
#include "symmetry/symmetry_batch_gpu.hpp"
#include "symmetry/symmetry_gpu.hpp"
#include "transpose/transpose_batch_gpu.hpp"

namespace spfft {

template <typename T>
ExecutionBatchGPU<T>::ExecutionBatchGPU(
    const int numThreads, SizeType batchSize, std::shared_ptr<Parameters> param,
    HostArray<std::complex<T>>& array1, HostArray<std::complex<T>>& array2,
    GPUArray<typename gpu::fft::ComplexType<T>::type>& gpuArray1,
    GPUArray<typename gpu::fft::ComplexType<T>::type>& gpuArray2,
    const std::shared_ptr<GPUArray<char>>& fftWorkBuffer)
    : stream_(false),
      externalStream_(nullptr),
      startEvent_(false),
      endEvent_(false),
      numThreads_(numThreads),
      batchSize_(batchSize),
      scalingFactor_(static_cast<T>(
          1.0 / static_cast<double>(param->dim_x() * param->dim_y() * param->dim_z()))),
      zStickSymmetry_(new Symmetry()),
      planeSymmetry_(new Symmetry()) {
  const SizeType numLocalZSticks = param->num_z_sticks(0);

  // frequency data with z-sticks: (batchSize * numZSticks, dimZ)
  freqDomainDataGPU_ =
      create_2d_view(gpuArray1, 0, batchSize * numLocalZSticks, param->dim_z());

  // compressed data buffer in gpuArray2 (used only during compress/decompress, not overlapping XY)
  freqDomainCompressedDataGPU_ = GPUArrayView1D<T>(
      reinterpret_cast<T*>(gpuArray2.data()),
      batchSize * param->local_value_indices().size() * 2, gpuArray2.device_id());

  // Z transform
  if (numLocalZSticks > 0) {
    transformZ_ = std::unique_ptr<TransformGPU>(
        new Transform1DGPU<T>(freqDomainDataGPU_, stream_, fftWorkBuffer));

    if (param->transform_type() == SPFFT_TRANS_R2C &&
        param->zero_zero_stick_index() < numLocalZSticks) {
      zStickSymmetry_.reset(new StickSymmetryBatchGPU<T>(
          stream_, freqDomainDataGPU_, param->zero_zero_stick_index(),
          numLocalZSticks, batchSize));
    }
  }

  if (numLocalZSticks > 0 && param->local_value_indices().size() > 0) {
    compression_.reset(new CompressionBatchGPU(param, batchSize));
  }

  // Transpose + XY domain: (batchSize * dimZ, dimY, dimX_freq)
  freqDomainXYGPU_ = create_3d_view(gpuArray2, 0, batchSize * param->dim_z(), param->dim_y(),
                                    param->dim_x_freq());
  transpose_.reset(new TransposeBatchGPU<T>(param, batchSize, stream_, freqDomainXYGPU_,
                                            freqDomainDataGPU_));

  // XY transform
  if (param->dim_z() > 0) {
    if (param->transform_type() == SPFFT_TRANS_R2C) {
      planeSymmetry_.reset(new PlaneSymmetryGPU<T>(stream_, freqDomainXYGPU_));

      spaceDomainDataExternalHost_ = create_new_type_3d_view<T>(
          array1, batchSize * param->dim_z(), param->dim_y(), param->dim_x());
      spaceDomainDataExternalGPU_ = create_new_type_3d_view<T>(
          gpuArray1, batchSize * param->dim_z(), param->dim_y(), param->dim_x());

      transformXY_ = std::unique_ptr<TransformGPU>(new TransformReal2DGPU<T>(
          spaceDomainDataExternalGPU_, freqDomainXYGPU_, stream_, fftWorkBuffer));
    } else {
      spaceDomainDataExternalHost_ = create_new_type_3d_view<T>(
          array1, batchSize * param->dim_z(), param->dim_y(), 2 * param->dim_x_freq());
      spaceDomainDataExternalGPU_ = create_new_type_3d_view<T>(
          freqDomainXYGPU_, batchSize * param->dim_z(), param->dim_y(),
          2 * param->dim_x_freq());

      transformXY_ = std::unique_ptr<TransformGPU>(
          new Transform2DGPU<T>(freqDomainXYGPU_, stream_, fftWorkBuffer));
    }
  }
}

template <typename T>
auto ExecutionBatchGPU<T>::forward_xy(const T* input) -> void {
  if (gpu::get_last_error() != gpu::status::Success) {
    throw GPUPrecedingError();
  }

  startEvent_.record(externalStream_);
  startEvent_.stream_wait(stream_.get());

  const T* inputPtrHost = nullptr;
  const T* inputPtrGPU = nullptr;
  std::tie(inputPtrHost, inputPtrGPU) = translate_gpu_pointer(input);
  if (!inputPtrGPU) inputPtrGPU = spaceDomainDataExternalGPU_.data();

  if (transformXY_) {
    if (inputPtrHost) {
      gpu::check_status(gpu::memcpy_async(static_cast<void*>(spaceDomainDataExternalGPU_.data()),
                                          static_cast<const void*>(inputPtrHost),
                                          spaceDomainDataExternalGPU_.size() * sizeof(T),
                                          gpu::flag::MemcpyHostToDevice, stream_.get()));
    }
    transformXY_->forward(inputPtrGPU, freqDomainXYGPU_.data());
  }

  if (transformXY_) transpose_->pack_forward();
}

template <typename T>
auto ExecutionBatchGPU<T>::forward_exchange(const bool nonBlockingExchange) -> void {
  transpose_->exchange_forward_start(nonBlockingExchange);
}

template <typename T>
auto ExecutionBatchGPU<T>::forward_z(T* output, const SpfftScalingType scalingType) -> void {
  transpose_->exchange_forward_finalize();

  if (transformZ_) transpose_->unpack_forward();

  if (transformZ_) transformZ_->forward();

  if (compression_) {
    T* outputPtrHost = nullptr;
    T* outputPtrGPU = nullptr;
    std::tie(outputPtrHost, outputPtrGPU) = translate_gpu_pointer(output);

    if (outputPtrGPU == nullptr) {
      compression_->compress(stream_, freqDomainDataGPU_, freqDomainCompressedDataGPU_.data(),
                             scalingType == SpfftScalingType::SPFFT_FULL_SCALING, scalingFactor_);

      gpu::check_status(gpu::memcpy_async(
          static_cast<void*>(outputPtrHost),
          static_cast<const void*>(freqDomainCompressedDataGPU_.data()),
          freqDomainCompressedDataGPU_.size() *
              sizeof(decltype(*(freqDomainCompressedDataGPU_.data()))),
          gpu::flag::MemcpyDeviceToHost, stream_.get()));
    } else {
      compression_->compress(stream_, freqDomainDataGPU_, outputPtrGPU,
                             scalingType == SpfftScalingType::SPFFT_FULL_SCALING, scalingFactor_);
    }
  }
}

template <typename T>
auto ExecutionBatchGPU<T>::backward_z(const T* input) -> void {
  if (gpu::get_last_error() != gpu::status::Success) {
    throw GPUPrecedingError();
  }

  startEvent_.record(externalStream_);
  startEvent_.stream_wait(stream_.get());

  if (compression_) {
    const T* inputPtrHost = nullptr;
    const T* inputPtrGPU = nullptr;
    std::tie(inputPtrHost, inputPtrGPU) = translate_gpu_pointer(input);

    startEvent_.record(nullptr);
    startEvent_.stream_wait(stream_.get());

    if (inputPtrGPU == nullptr) {
      gpu::check_status(gpu::memcpy_async(
          static_cast<void*>(freqDomainCompressedDataGPU_.data()),
          static_cast<const void*>(inputPtrHost),
          freqDomainCompressedDataGPU_.size() *
              sizeof(decltype(*(freqDomainCompressedDataGPU_.data()))),
          gpu::flag::MemcpyHostToDevice, stream_.get()));
      compression_->decompress(stream_, freqDomainCompressedDataGPU_.data(), freqDomainDataGPU_);
    } else {
      compression_->decompress(stream_, inputPtrGPU, freqDomainDataGPU_);
    }
  }

  if (transformZ_) {
    zStickSymmetry_->apply();
    transformZ_->backward();
  }

  if (transformZ_) transpose_->pack_backward();
}

template <typename T>
auto ExecutionBatchGPU<T>::backward_exchange(const bool nonBlockingExchange) -> void {
  transpose_->exchange_backward_start(nonBlockingExchange);
}

template <typename T>
auto ExecutionBatchGPU<T>::backward_xy(T* output) -> void {
  transpose_->exchange_backward_finalize();

  T* outputPtrHost = nullptr;
  T* outputPtrGPU = nullptr;
  std::tie(outputPtrHost, outputPtrGPU) = translate_gpu_pointer(output);
  if (!outputPtrGPU) outputPtrGPU = spaceDomainDataExternalGPU_.data();

  if (transformXY_) {
    transpose_->unpack_backward();
    planeSymmetry_->apply();
    transformXY_->backward(freqDomainXYGPU_.data(), outputPtrGPU);
    if (outputPtrHost) {
      gpu::check_status(
          gpu::memcpy_async(static_cast<void*>(outputPtrHost),
                            static_cast<const void*>(spaceDomainDataExternalGPU_.data()),
                            spaceDomainDataExternalGPU_.size() * sizeof(T),
                            gpu::flag::MemcpyDeviceToHost, stream_.get()));
    }
  }
}

template <typename T>
auto ExecutionBatchGPU<T>::synchronize(SpfftExecType mode) -> void {
  if (mode == SPFFT_EXEC_ASYNCHRONOUS) {
    endEvent_.record(stream_.get());
    endEvent_.stream_wait(externalStream_);
  } else {
    gpu::check_status(gpu::stream_synchronize(stream_.get()));
  }
}

template <typename T>
auto ExecutionBatchGPU<T>::space_domain_data_host() -> HostArrayView3D<T> {
  return spaceDomainDataExternalHost_;
}

template <typename T>
auto ExecutionBatchGPU<T>::space_domain_data_gpu() -> GPUArrayView3D<T> {
  return spaceDomainDataExternalGPU_;
}

template class ExecutionBatchGPU<double>;
#ifdef SPFFT_SINGLE_PRECISION
template class ExecutionBatchGPU<float>;
#endif

}  // namespace spfft
