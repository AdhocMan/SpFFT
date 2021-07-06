/*
 * Copyright (c) 2019 ETH Zurich, Simon Frasch
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
#include <algorithm>
#include <cassert>
#include <complex>
#include <cstring>
#include <utility>
#include <vector>
#include "memory/array_view_utility.hpp"
#include "memory/host_array_view.hpp"
#include "parameters/parameters.hpp"
#include "spfft/exceptions.hpp"
#include "transpose.hpp"
#include "util/common_types.hpp"
#include "util/omp_definitions.hpp"
#include "util/type_check.hpp"

#if defined(SPFFT_MPI) && defined(SPFFT_COSTA)
#include "mpi_util/mpi_check_status.hpp"
#include "mpi_util/mpi_communicator_handle.hpp"
#include "mpi_util/mpi_datatype_handle.hpp"
#include "mpi_util/mpi_match_elementary_type.hpp"
#include "transpose/transpose_mpi_costa_host.hpp"

namespace spfft {
template <typename T>
TransposeMPICostaHost<T>::TransposeMPICostaHost(
    const std::shared_ptr<Parameters>& param, MPICommunicatorHandle comm,
    HostArrayView3D<ComplexType> spaceDomainData, HostArrayView2D<ComplexType> freqDomainData)
    : comm_(std::move(comm)),
      spaceDomainData_(spaceDomainData),
      freqDomainData_(freqDomainData),
      costaTransf_(comm_.get()) {
  assert(disjoint(spaceDomainData, freqDomainData));
  assert(param->dim_x_freq() == spaceDomainData.dim_mid());
  assert(param->dim_y() == spaceDomainData.dim_inner());
  assert(param->num_xy_planes(comm_.rank()) == spaceDomainData.dim_outer());
  assert(param->dim_z() == freqDomainData.dim_inner());
  assert(param->num_z_sticks(comm_.rank()) == freqDomainData.dim_outer());

  // Create layout for freq domain
  {
    std::vector<int> rowSplit = {0, static_cast<int>(param->dim_z())};
    std::vector<int> owners;
    owners.reserve(comm_.size());
    std::vector<int> colSplit;
    colSplit.reserve(comm_.size());

    colSplit.push_back(0);
    SizeType localBlockIdx = 0;
    for (SizeType r = 0; r < comm_.size(); ++r) {
      if (param->num_z_sticks(r)) {
        owners.push_back(r);
        colSplit.push_back(colSplit.back() + param->num_z_sticks(r));
        if (r == comm_.rank()) localBlockIdx = owners.size() - 1;
      }
    }

    costa::block_t localBlock{freqDomainData_.data(), static_cast<int>(param->dim_z()), 0,
                              static_cast<int>(localBlockIdx)};

    freqDomainLayout_.reset(
        new costa::grid_layout<std::complex<T>>(costa::custom_layout<std::complex<T>>(
            rowSplit.size() - 1, colSplit.size() - 1, rowSplit.data(), colSplit.data(),
            owners.data(), param->num_z_sticks(comm_.rank()) > 0, &localBlock, 'C')));
  }

  // Create layout for space domain
  {
    SizeType numGlobalZSticks = 0;
    for(SizeType r =0; r < comm_.size(); ++r) {
      numGlobalZSticks += param->num_z_sticks(r);
    }

    std::vector<int> rowSplit;
    rowSplit.reserve(comm_.size());
    std::vector<int> owners;
    owners.reserve(numGlobalZSticks * comm_.size());
    std::vector<int> colSplit;
    colSplit.reserve(numGlobalZSticks);

    rowSplit.push_back(0);
    SizeType localRowBlockIdx = 0;
    for(SizeType r =0; r < comm_.size(); ++r) {
      if(param->num_xy_planes(r)) {
        owners.insert(owners.end(), numGlobalZSticks, static_cast<int>(r));
        rowSplit.push_back(rowSplit.back() + param->num_xy_planes(r));
        if (r == comm_.rank()) {
          localRowBlockIdx = rowSplit.size() - 1;
        }
      }
    }

    colSplit.push_back(0);
    for(SizeType i = 0; i < numGlobalZSticks; ++i) {
      colSplit.push_back(i + 1);
    }

    std::vector<costa::block_t> localBlocks;
    if(param->num_xy_planes(comm_.rank())) {
      localBlocks.reserve(numGlobalZSticks);
      int counter = 0;
      for (SizeType r = 0; r < comm_.size(); ++r) {
        const auto zStickXYIndices = param->z_stick_xy_indices(r);
        for(auto& index : zStickXYIndices) {
          localBlocks.push_back({spaceDomainData_.data() + index,
                                 static_cast<int>(param->dim_x() * param->dim_y()),
                                 static_cast<int>(localRowBlockIdx), counter});
          ++counter;
        }
      }
    }

    spaceDomainLayout_.reset(
        new costa::grid_layout<std::complex<T>>(costa::custom_layout<std::complex<T>>(
            rowSplit.size() - 1, colSplit.size() - 1, rowSplit.data(), colSplit.data(),
            owners.data(), param->num_xy_planes(comm_.rank()) > 0, localBlocks.data(), 'R')));
  }
}

template <typename T>
auto TransposeMPICostaHost<T>::exchange_backward_start(const bool nonBlockingExchange)
    -> void {
  assert(omp_get_thread_num() == 0);  // only must thread must be allowed to enter

  // zero target data location (not all values are overwritten upon unpacking)
  std::memset(static_cast<void*>(spaceDomainData_.data()), 0,
              sizeof(typename decltype(spaceDomainData_)::ValueType) * spaceDomainData_.size());

  costaTransf_.schedule(*freqDomainLayout_, *spaceDomainLayout_, 'N', 1.0, 0.0);
  costaTransf_.transform();
}

template <typename T>
auto TransposeMPICostaHost<T>::exchange_backward_finalize() -> void {
}

template <typename T>
auto TransposeMPICostaHost<T>::exchange_forward_start(const bool nonBlockingExchange) -> void {
  assert(omp_get_thread_num() == 0);  // only must thread must be allowed to enter

  costaTransf_.schedule(*spaceDomainLayout_, *freqDomainLayout_, 'N', 1.0, 0.0);
  costaTransf_.transform();
}

template <typename T>
auto TransposeMPICostaHost<T>::exchange_forward_finalize() -> void {
}

// Instantiate class for float and double
#ifdef SPFFT_SINGLE_PRECISION
template class TransposeMPICostaHost<float>;
#endif
template class TransposeMPICostaHost<double>;
}  // namespace spfft
#endif  // SPFFT_MPI
