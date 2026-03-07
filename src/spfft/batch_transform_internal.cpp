#include "spfft/batch_transform_internal.hpp"
#include "spfft/exceptions.hpp"
#include "util/omp_definitions.hpp"
#include "timing/timing.hpp"

namespace spfft {

template <typename T>
BatchTransformInternal<T>::BatchTransformInternal(int maxNumThreads,
                                                   SpfftProcessingUnitType processingUnit,
                                                   SpfftTransformType transformType, int dimX,
                                                   int dimY, int dimZ, int batchSize,
                                                   int numLocalElements,
                                                   SpfftIndexFormatType indexFormat,
                                                   const int* indices)
    : batchSize_(batchSize), numThreads_(maxNumThreads), processingUnit_(processingUnit) {
  if (dimX < 0 || dimY < 0 || dimZ < 0 || batchSize < 1 || numLocalElements < 0 ||
      (!indices && numLocalElements > 0)) {
    throw InvalidParameterError();
  }

  if (maxNumThreads < 1) {
    numThreads_ = omp_get_max_threads();
  }

  param_.reset(
      new Parameters(transformType, dimX, dimY, dimZ, numLocalElements, indexFormat, indices));

  const SizeType numLocalZSticks = param_->num_z_sticks(0);
  const SizeType array1SizeActual =
      static_cast<SizeType>(batchSize) *
      std::max(numLocalZSticks * static_cast<SizeType>(dimZ),
               static_cast<SizeType>(dimX) * static_cast<SizeType>(dimY) *
                   static_cast<SizeType>(dimZ));
  const SizeType array2Size =
      static_cast<SizeType>(batchSize) * static_cast<SizeType>(dimZ) *
      param_->dim_x_freq() * static_cast<SizeType>(dimY);

  arrayHost1_ = HostArray<std::complex<T>>(array1SizeActual);
  arrayHost2_ = HostArray<std::complex<T>>(array2Size);

  if (processingUnit == SPFFT_PU_HOST) {
    execHost_.reset(
        new ExecutionBatchHost<T>(numThreads_, batchSize, param_, arrayHost1_, arrayHost2_));
  } else {
#if defined(SPFFT_CUDA) || defined(SPFFT_ROCM)
    gpuArray1_ = GPUArray<typename gpu::fft::ComplexType<T>::type>(array1SizeActual);
    gpuArray2_ = GPUArray<typename gpu::fft::ComplexType<T>::type>(array2Size);
    fftWorkBuffer_ = std::make_shared<GPUArray<char>>(0);
    execGPU_.reset(new ExecutionBatchGPU<T>(numThreads_, batchSize, param_, arrayHost1_,
                                            arrayHost2_, gpuArray1_, gpuArray2_, fftWorkBuffer_));
#else
    throw GPUSupportError();
#endif
  }
}

template <typename T>
auto BatchTransformInternal<T>::forward(const T* input, T* output, SpfftScalingType scaling)
    -> void {
  HOST_TIMING_SCOPED("forward")
  if (processingUnit_ == SPFFT_PU_HOST) {
    execHost_->forward_xy(input);
    execHost_->forward_exchange(false);
    execHost_->forward_z(output, scaling);
  } else {
#if defined(SPFFT_CUDA) || defined(SPFFT_ROCM)
    execGPU_->forward_xy(input);
    execGPU_->forward_exchange(false);
    execGPU_->forward_z(output, scaling);
    execGPU_->synchronize(SPFFT_EXEC_SYNCHRONOUS);
#endif
  }
}

template <typename T>
auto BatchTransformInternal<T>::forward(SpfftProcessingUnitType inputLocation, T* output,
                                         SpfftScalingType scaling) -> void {
  if (processingUnit_ == SPFFT_PU_HOST && inputLocation != SPFFT_PU_HOST) {
    throw InvalidParameterError();
  }
  this->forward(this->space_domain_data(inputLocation), output, scaling);
}

template <typename T>
auto BatchTransformInternal<T>::backward(const T* input, T* output) -> void {
  HOST_TIMING_SCOPED("backward")
  if (processingUnit_ == SPFFT_PU_HOST) {
    execHost_->backward_z(input);
    execHost_->backward_exchange(false);
    execHost_->backward_xy(output);
  } else {
#if defined(SPFFT_CUDA) || defined(SPFFT_ROCM)
    execGPU_->backward_z(input);
    execGPU_->backward_exchange(false);
    execGPU_->backward_xy(output);
    execGPU_->synchronize(SPFFT_EXEC_SYNCHRONOUS);
#endif
  }
}

template <typename T>
auto BatchTransformInternal<T>::backward(const T* input, SpfftProcessingUnitType outputLocation)
    -> void {
  if (processingUnit_ == SPFFT_PU_HOST && outputLocation != SPFFT_PU_HOST) {
    throw InvalidParameterError();
  }
  this->backward(input, this->space_domain_data(outputLocation));
}

template <typename T>
auto BatchTransformInternal<T>::space_domain_data(SpfftProcessingUnitType pu) -> T* {
  if (pu == SPFFT_PU_HOST) {
    if (processingUnit_ == SPFFT_PU_HOST) {
      return execHost_->space_domain_data().data();
    } else {
#if defined(SPFFT_CUDA) || defined(SPFFT_ROCM)
      return execGPU_->space_domain_data_host().data();
#else
      throw GPUSupportError();
#endif
    }
  } else {
#if defined(SPFFT_CUDA) || defined(SPFFT_ROCM)
    return execGPU_->space_domain_data_gpu().data();
#else
    throw GPUSupportError();
#endif
  }
}

template class BatchTransformInternal<double>;
#ifdef SPFFT_SINGLE_PRECISION
template class BatchTransformInternal<float>;
#endif

}  // namespace spfft
