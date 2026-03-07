#ifndef SPFFT_TRANSPOSE_BATCH_HOST_HPP
#define SPFFT_TRANSPOSE_BATCH_HOST_HPP

#include <cassert>
#include <complex>
#include <cstring>
#include <memory>
#include "memory/array_view_utility.hpp"
#include "memory/host_array_view.hpp"
#include "parameters/parameters.hpp"
#include "spfft/config.h"
#include "transpose/transpose.hpp"
#include "util/common_types.hpp"
#include "util/omp_definitions.hpp"
#include "util/type_check.hpp"

namespace spfft {

template <typename T>
class TransposeBatchHost : public Transpose {
  static_assert(IsFloatOrDouble<T>::value, "Type T must be float or double");
  using ComplexType = std::complex<T>;

public:
  TransposeBatchHost(const std::shared_ptr<Parameters>& param, SizeType batchSize,
                     HostArrayView3D<ComplexType> spaceDomainData,
                     HostArrayView2D<ComplexType> freqDomainData)
      : spaceDomainData_(spaceDomainData),
        freqDomainData_(freqDomainData),
        param_(param),
        batchSize_(batchSize),
        numZSticks_(param->num_z_sticks(0)),
        dimZ_(param->dim_z()) {
    assert(spaceDomainData.dim_outer() == batchSize * param->dim_z());
    assert(freqDomainData.dim_outer() == batchSize * numZSticks_);
    assert(freqDomainData.dim_inner() == dimZ_);
    assert(disjoint(spaceDomainData, freqDomainData));
  }

  auto exchange_backward_start(const bool) -> void override {}
  auto exchange_forward_start(const bool) -> void override {}

  auto unpack_backward() -> void override {
    SPFFT_OMP_PRAGMA("omp for schedule(static)")
    for (SizeType z = 0; z < spaceDomainData_.dim_outer(); ++z) {
      std::memset(static_cast<void*>(&spaceDomainData_(z, 0, 0)), 0,
                  sizeof(ComplexType) * spaceDomainData_.dim_inner() * spaceDomainData_.dim_mid());
    }

    auto stickIndicesView = param_->z_stick_xy_indices(0);
    auto spaceDomainDataFlat =
        create_2d_view(spaceDomainData_, 0, spaceDomainData_.dim_outer(),
                       spaceDomainData_.dim_mid() * spaceDomainData_.dim_inner());

    const SizeType totalWork = batchSize_ * numZSticks_;

    SPFFT_OMP_PRAGMA("omp for schedule(static)")
    for (SizeType linearIdx = 0; linearIdx < totalWork; ++linearIdx) {
      const SizeType b = linearIdx / numZSticks_;
      const SizeType zStickIndex = linearIdx % numZSticks_;
      const SizeType xyIndex = stickIndicesView(zStickIndex);
      const SizeType bFreqOffset = b * numZSticks_;
      const SizeType bSpaceOffset = b * dimZ_;
      for (SizeType zIndex = 0; zIndex < dimZ_; ++zIndex) {
        spaceDomainDataFlat(bSpaceOffset + zIndex, xyIndex) =
            freqDomainData_(bFreqOffset + zStickIndex, zIndex);
      }
    }
  }

  auto unpack_forward() -> void override {
    auto stickIndicesView = param_->z_stick_xy_indices(0);
    auto spaceDomainDataFlat =
        create_2d_view(spaceDomainData_, 0, spaceDomainData_.dim_outer(),
                       spaceDomainData_.dim_mid() * spaceDomainData_.dim_inner());

    const SizeType totalWork = batchSize_ * numZSticks_;

    SPFFT_OMP_PRAGMA("omp for schedule(static)")
    for (SizeType linearIdx = 0; linearIdx < totalWork; ++linearIdx) {
      const SizeType b = linearIdx / numZSticks_;
      const SizeType zStickIndex = linearIdx % numZSticks_;
      const SizeType xyIndex = stickIndicesView(zStickIndex);
      const SizeType bFreqOffset = b * numZSticks_;
      const SizeType bSpaceOffset = b * dimZ_;
      for (SizeType zIndex = 0; zIndex < dimZ_; ++zIndex) {
        freqDomainData_(bFreqOffset + zStickIndex, zIndex) =
            spaceDomainDataFlat(bSpaceOffset + zIndex, xyIndex);
      }
    }
  }

private:
  HostArrayView3D<ComplexType> spaceDomainData_;
  HostArrayView2D<ComplexType> freqDomainData_;
  std::shared_ptr<Parameters> param_;
  SizeType batchSize_;
  SizeType numZSticks_;
  SizeType dimZ_;
};
}  // namespace spfft
#endif
