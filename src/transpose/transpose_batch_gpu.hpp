#ifndef SPFFT_TRANSPOSE_BATCH_GPU_HPP
#define SPFFT_TRANSPOSE_BATCH_GPU_HPP

#include <algorithm>
#include <cassert>
#include <complex>
#include <memory>
#include <vector>
#include "gpu_util/gpu_fft_api.hpp"
#include "gpu_util/gpu_stream_handle.hpp"
#include "memory/array_view_utility.hpp"
#include "memory/gpu_array.hpp"
#include "memory/gpu_array_view.hpp"
#include "parameters/parameters.hpp"
#include "spfft/config.h"
#include "transpose/gpu_kernels/local_transpose_batch_kernels.hpp"
#include "transpose/transpose.hpp"
#include "util/common_types.hpp"
#include "util/type_check.hpp"

namespace spfft {

template <typename T>
class TransposeBatchGPU : public Transpose {
  static_assert(IsFloatOrDouble<T>::value, "Type T must be float or double");
  using ComplexType = typename gpu::fft::ComplexType<T>::type;

public:
  TransposeBatchGPU(const std::shared_ptr<Parameters>& param, SizeType batchSize,
                    GPUStreamHandle stream, GPUArrayView3D<ComplexType> spaceDomainData,
                    GPUArrayView2D<ComplexType> freqDomainData)
      : stream_(std::move(stream)),
        spaceDomainData_(spaceDomainData),
        freqDomainData_(freqDomainData),
        batchSize_(batchSize),
        numZSticks_(param->num_z_sticks(0)),
        dimZ_(param->dim_z()),
        indices_(param->num_z_sticks(0)) {
    assert(spaceDomainData.dim_outer() == batchSize * param->dim_z());
    assert(freqDomainData.dim_outer() == batchSize * numZSticks_);
    assert(freqDomainData.dim_inner() == dimZ_);

    assert(disjoint(spaceDomainData, freqDomainData));

    const auto zStickXYIndices = param->z_stick_xy_indices(0);

    std::vector<int> transposedIndices;
    transposedIndices.reserve(zStickXYIndices.size());

    for (const auto& index : zStickXYIndices) {
      const int x = index / param->dim_y();
      const int y = index - x * param->dim_y();
      transposedIndices.emplace_back(y * param->dim_x_freq() + x);
    }

    copy_to_gpu(transposedIndices, indices_);
  }

  auto exchange_backward_start(const bool) -> void override {
    gpu::check_status(gpu::memset_async(
        static_cast<void*>(spaceDomainData_.data()), 0,
        spaceDomainData_.size() * sizeof(typename decltype(spaceDomainData_)::ValueType),
        stream_.get()));

    if (freqDomainData_.size() > 0 && spaceDomainData_.size() > 0) {
      const auto indicesView = create_1d_view(indices_, 0, indices_.size());
      const int xyPlaneSize = spaceDomainData_.dim_mid() * spaceDomainData_.dim_inner();
      local_transpose_batch_backward(stream_.get(), indicesView, freqDomainData_.data(),
                                     spaceDomainData_.data(), static_cast<int>(numZSticks_),
                                     static_cast<int>(dimZ_), xyPlaneSize,
                                     static_cast<int>(batchSize_));
    }
  }

  auto unpack_backward() -> void override {}

  auto exchange_forward_start(const bool) -> void override {
    if (freqDomainData_.size() > 0 && spaceDomainData_.size() > 0) {
      const auto indicesView = create_1d_view(indices_, 0, indices_.size());
      const int xyPlaneSize = spaceDomainData_.dim_mid() * spaceDomainData_.dim_inner();
      local_transpose_batch_forward(stream_.get(), indicesView, spaceDomainData_.data(),
                                    freqDomainData_.data(), static_cast<int>(numZSticks_),
                                    static_cast<int>(dimZ_), xyPlaneSize,
                                    static_cast<int>(batchSize_));
    }
  }

  auto unpack_forward() -> void override {}

private:
  GPUStreamHandle stream_;
  GPUArrayView3D<ComplexType> spaceDomainData_;
  GPUArrayView2D<ComplexType> freqDomainData_;
  SizeType batchSize_;
  SizeType numZSticks_;
  SizeType dimZ_;
  GPUArray<int> indices_;
};
}  // namespace spfft
#endif
