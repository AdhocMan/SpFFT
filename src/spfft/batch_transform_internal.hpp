#ifndef SPFFT_BATCH_TRANSFORM_INTERNAL_HPP
#define SPFFT_BATCH_TRANSFORM_INTERNAL_HPP

#include <memory>
#include "execution/execution_batch_host.hpp"
#include "memory/host_array.hpp"
#include "parameters/parameters.hpp"
#include "spfft/config.h"
#include "spfft/types.h"
#include "util/common_types.hpp"

namespace spfft {
template <typename T>
class BatchTransformInternal {
public:
  BatchTransformInternal(int maxNumThreads, SpfftTransformType transformType, int dimX, int dimY,
                         int dimZ, int batchSize, int numLocalElements,
                         SpfftIndexFormatType indexFormat, const int* indices);

  auto forward(const T* input, T* output, SpfftScalingType scaling) -> void;
  auto forward(T* output, SpfftScalingType scaling) -> void;

  auto backward(const T* input, T* output) -> void;
  auto backward(const T* input) -> void;

  auto space_domain_data() -> T*;

  inline auto batch_size() const noexcept -> int { return batchSize_; }
  inline auto dim_x() const noexcept -> int { return param_->dim_x(); }
  inline auto dim_y() const noexcept -> int { return param_->dim_y(); }
  inline auto dim_z() const noexcept -> int { return param_->dim_z(); }
  inline auto num_local_elements() const noexcept -> int { return param_->local_num_elements(); }
  inline auto type() const noexcept -> SpfftTransformType { return param_->transform_type(); }
  inline auto num_threads() const noexcept -> int { return numThreads_; }

private:
  int batchSize_;
  int numThreads_;
  std::shared_ptr<Parameters> param_;
  HostArray<std::complex<T>> arrayHost1_;
  HostArray<std::complex<T>> arrayHost2_;
  std::unique_ptr<ExecutionBatchHost<T>> execHost_;
};

}  // namespace spfft

#endif
