#ifndef SPFFT_EXECUTION_BATCH_HOST_HPP
#define SPFFT_EXECUTION_BATCH_HOST_HPP

#include <complex>
#include <memory>
#include "compression/compression_batch_host.hpp"
#include "fft/transform_interface.hpp"
#include "memory/host_array.hpp"
#include "memory/host_array_view.hpp"
#include "parameters/parameters.hpp"
#include "spfft/config.h"
#include "spfft/types.h"
#include "symmetry/symmetry.hpp"
#include "transpose/transpose.hpp"
#include "util/common_types.hpp"
#include "util/omp_definitions.hpp"

namespace spfft {

template <typename T>
class ExecutionBatchHost {
public:
  ExecutionBatchHost(const int numThreads, SizeType batchSize, std::shared_ptr<Parameters> param,
                     HostArray<std::complex<T>>& array1, HostArray<std::complex<T>>& array2);

  auto forward_z(T* output, const SpfftScalingType scalingType) -> void;
  auto forward_exchange(const bool nonBlockingExchange) -> void;
  auto forward_xy(const T* input) -> void;

  auto backward_z(const T* input) -> void;
  auto backward_exchange(const bool nonBlockingExchange) -> void;
  auto backward_xy(T* output) -> void;

  auto space_domain_data() -> HostArrayView3D<T>;

private:
  int numThreads_;
  SizeType batchSize_;
  T scalingFactor_;
  std::unique_ptr<TransformHost<T>> transformZBackward_;
  std::unique_ptr<TransformHost<T>> transformZForward_;
  std::unique_ptr<TransformHost<T>> transformYBackward_;
  std::unique_ptr<TransformHost<T>> transformYForward_;
  std::unique_ptr<TransformHost<T>> transformXBackward_;
  std::unique_ptr<TransformHost<T>> transformXForward_;

  std::unique_ptr<Transpose> transpose_;

  std::unique_ptr<Symmetry> zStickSymmetry_;
  std::unique_ptr<Symmetry> planeSymmetry_;

  std::unique_ptr<CompressionBatchHost> compression_;

  HostArrayView3D<T> spaceDomainDataExternal_;
  HostArrayView2D<std::complex<T>> freqDomainData_;
  HostArrayView3D<std::complex<T>> freqDomainXY_;
};
}  // namespace spfft
#endif
