#include "execution/execution_batch_host.hpp"
#include "fft/transform_1d_host.hpp"
#include "fft/transform_real_1d_host.hpp"
#include "memory/array_view_utility.hpp"
#include "memory/host_array_view.hpp"
#include "spfft/exceptions.hpp"
#include "symmetry/symmetry_batch_host.hpp"
#include "symmetry/symmetry_host.hpp"
#include "transpose/transpose_batch_host.hpp"
#include "util/common_types.hpp"

namespace spfft {

template <typename T>
ExecutionBatchHost<T>::ExecutionBatchHost(const int numThreads, SizeType batchSize,
                                          std::shared_ptr<Parameters> param,
                                          HostArray<std::complex<T>>& array1,
                                          HostArray<std::complex<T>>& array2)
    : numThreads_(numThreads),
      batchSize_(batchSize),
      scalingFactor_(static_cast<T>(
          1.0 / static_cast<double>(param->dim_x() * param->dim_y() * param->dim_z()))),
      zStickSymmetry_(new Symmetry()),
      planeSymmetry_(new Symmetry()) {
  const SizeType numLocalZSticks = param->num_z_sticks(0);
  std::set<SizeType> uniqueXIndices;
  for (const auto& xyIndex : param->z_stick_xy_indices(0)) {
    uniqueXIndices.emplace(static_cast<SizeType>(xyIndex / param->dim_y()));
  }

  auto freqDomainZ3D =
      create_3d_view(array1, 0, batchSize, numLocalZSticks, param->dim_z());
  freqDomainData_ =
      create_2d_view(freqDomainZ3D, 0, batchSize * numLocalZSticks, param->dim_z());
  freqDomainXY_ =
      create_3d_view(array2, 0, batchSize * param->dim_z(), param->dim_x_freq(), param->dim_y());

  transpose_.reset(
      new TransposeBatchHost<T>(param, batchSize, freqDomainXY_, freqDomainData_));

  if (param->local_value_indices().size() > 0) {
    compression_.reset(new CompressionBatchHost(param, batchSize));
  }

  if (numLocalZSticks > 0) {
    transformZBackward_.reset(new Transform1DPlanesHost<T>(freqDomainZ3D, freqDomainZ3D, false,
                                                           false, FFTW_BACKWARD, numThreads));
    transformZForward_.reset(new Transform1DPlanesHost<T>(freqDomainZ3D, freqDomainZ3D, false,
                                                          false, FFTW_FORWARD, numThreads));
  }

  if (param->dim_z() > 0) {
    transformYBackward_.reset(new Transform1DVerticalHost<T>(freqDomainXY_, freqDomainXY_, false,
                                                             false, FFTW_BACKWARD, uniqueXIndices));
    transformYForward_.reset(new Transform1DVerticalHost<T>(freqDomainXY_, freqDomainXY_, false,
                                                            false, FFTW_FORWARD, uniqueXIndices));

    if (param->transform_type() == SPFFT_TRANS_R2C) {
      if (param->zero_zero_stick_index() < param->num_z_sticks(0)) {
        zStickSymmetry_.reset(new StickSymmetryBatchHost<T>(
            freqDomainData_, param->zero_zero_stick_index(), numLocalZSticks, batchSize));
      }

      planeSymmetry_.reset(new PlaneSymmetryHost<T>(freqDomainXY_));

      spaceDomainDataExternal_ = create_new_type_3d_view<T>(
          array1, batchSize * param->dim_z(), param->dim_y(), param->dim_x());
      transformXBackward_.reset(new C2RTransform1DPlanesHost<T>(
          freqDomainXY_, spaceDomainDataExternal_, true, false, numThreads));
      transformXForward_.reset(new R2CTransform1DPlanesHost<T>(
          spaceDomainDataExternal_, freqDomainXY_, false, true, numThreads));
    } else {
      auto spaceDomainData = create_3d_view(
          array1, 0, batchSize * param->dim_z(), param->dim_y(), param->dim_x_freq());
      spaceDomainDataExternal_ = create_new_type_3d_view<T>(
          array1, batchSize * param->dim_z(), param->dim_y(), 2 * param->dim_x());
      transformXBackward_.reset(new Transform1DPlanesHost<T>(freqDomainXY_, spaceDomainData, true,
                                                             false, FFTW_BACKWARD, numThreads));
      transformXForward_.reset(new Transform1DPlanesHost<T>(spaceDomainData, freqDomainXY_, false,
                                                            true, FFTW_FORWARD, numThreads));
    }
  }
}

template <typename T>
auto ExecutionBatchHost<T>::forward_xy(const T* input) -> void {
  SPFFT_OMP_PRAGMA("omp parallel num_threads(numThreads_)") {
    if (transformXForward_)
      transformXForward_->execute(input, reinterpret_cast<T*>(freqDomainXY_.data()));

    if (transformYForward_) transformYForward_->execute();

    if (transformYForward_) transpose_->pack_forward();
  }
}

template <typename T>
auto ExecutionBatchHost<T>::forward_exchange(const bool nonBlockingExchange) -> void {
  transpose_->exchange_forward_start(nonBlockingExchange);
}

template <typename T>
auto ExecutionBatchHost<T>::forward_z(T* output, const SpfftScalingType scalingType) -> void {
  transpose_->exchange_forward_finalize();

  SPFFT_OMP_PRAGMA("omp parallel num_threads(numThreads_)") {
    if (transformZForward_) transpose_->unpack_forward();

    if (transformZForward_) transformZForward_->execute();

    if (compression_)
      compression_->compress(freqDomainData_, output,
                             scalingType == SpfftScalingType::SPFFT_FULL_SCALING, scalingFactor_);
  }
}

template <typename T>
auto ExecutionBatchHost<T>::backward_z(const T* input) -> void {
  SPFFT_OMP_PRAGMA("omp parallel num_threads(numThreads_)") {
    if (compression_) compression_->decompress(input, freqDomainData_);

    zStickSymmetry_->apply();

    if (transformZBackward_) transformZBackward_->execute();

    if (transformZBackward_) transpose_->pack_backward();
  }
}

template <typename T>
auto ExecutionBatchHost<T>::backward_exchange(const bool nonBlockingExchange) -> void {
  transpose_->exchange_backward_start(nonBlockingExchange);
}

template <typename T>
auto ExecutionBatchHost<T>::backward_xy(T* output) -> void {
  transpose_->exchange_forward_finalize();

  SPFFT_OMP_PRAGMA("omp parallel num_threads(numThreads_)") {
    if (transformYBackward_) transpose_->unpack_backward();

    planeSymmetry_->apply();

    if (transformYBackward_) transformYBackward_->execute();

    if (transformXBackward_)
      transformXBackward_->execute(reinterpret_cast<T*>(freqDomainXY_.data()), output);
  }
}

template <typename T>
auto ExecutionBatchHost<T>::space_domain_data() -> HostArrayView3D<T> {
  return spaceDomainDataExternal_;
}

template class ExecutionBatchHost<double>;

#ifdef SPFFT_SINGLE_PRECISION
template class ExecutionBatchHost<float>;
#endif

}  // namespace spfft
