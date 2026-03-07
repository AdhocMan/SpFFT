#include "spfft/batch_transform_internal.hpp"
#include "spfft/exceptions.hpp"
#include "util/omp_definitions.hpp"

namespace spfft {

template <typename T>
BatchTransformInternal<T>::BatchTransformInternal(int maxNumThreads,
                                                   SpfftTransformType transformType, int dimX,
                                                   int dimY, int dimZ, int batchSize,
                                                   int numLocalElements,
                                                   SpfftIndexFormatType indexFormat,
                                                   const int* indices)
    : batchSize_(batchSize), numThreads_(maxNumThreads) {
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
  const SizeType array1Size =
      static_cast<SizeType>(batchSize) *
      std::max(numLocalZSticks * static_cast<SizeType>(dimZ),
               static_cast<SizeType>(dimX) * static_cast<SizeType>(dimY) *
                   static_cast<SizeType>(dimZ) / 2 + 1);
  // array1 needs to hold B * dimX * dimY * dimZ doubles reinterpreted as complex
  // For R2C: B * dimZ * dimY * dimX doubles => B * dimZ * dimY * dimX / 2 complex (rounded up)
  // For C2C: B * dimZ * dimY * 2*dimX doubles => B * dimZ * dimY * dimX complex (as dim_x_freq)
  // Simplify: use max of both needs
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

  execHost_.reset(
      new ExecutionBatchHost<T>(numThreads_, batchSize, param_, arrayHost1_, arrayHost2_));
}

template <typename T>
auto BatchTransformInternal<T>::forward(const T* input, T* output, SpfftScalingType scaling)
    -> void {
  execHost_->forward_xy(input);
  execHost_->forward_exchange(false);
  execHost_->forward_z(output, scaling);
}

template <typename T>
auto BatchTransformInternal<T>::forward(T* output, SpfftScalingType scaling) -> void {
  this->forward(this->space_domain_data(), output, scaling);
}

template <typename T>
auto BatchTransformInternal<T>::backward(const T* input, T* output) -> void {
  execHost_->backward_z(input);
  execHost_->backward_exchange(false);
  execHost_->backward_xy(output);
}

template <typename T>
auto BatchTransformInternal<T>::backward(const T* input) -> void {
  this->backward(input, this->space_domain_data());
}

template <typename T>
auto BatchTransformInternal<T>::space_domain_data() -> T* {
  return execHost_->space_domain_data().data();
}

template class BatchTransformInternal<double>;
#ifdef SPFFT_SINGLE_PRECISION
template class BatchTransformInternal<float>;
#endif

}  // namespace spfft
