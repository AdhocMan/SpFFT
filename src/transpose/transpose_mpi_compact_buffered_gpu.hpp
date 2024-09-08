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
#ifndef SPFFT_TRANSPOSE_MPI_COMPACT_BUFFERED_GPU_HPP
#define SPFFT_TRANSPOSE_MPI_COMPACT_BUFFERED_GPU_HPP

#include <complex>
#include <memory>
#include <vector>

#include "gpu_util/gpu_event_handle.hpp"
#include "gpu_util/gpu_fft_api.hpp"
#include "gpu_util/gpu_mem_view.hpp"
#include "gpu_util/gpu_stream_handle.hpp"
#include "memory/gpu_array.hpp"
#include "memory/gpu_array_view.hpp"
#include "memory/host_array_view.hpp"
#include "parameters/parameters.hpp"
#include "spfft/config.h"
#include "transpose.hpp"
#include "util/common_types.hpp"
#include "util/type_check.hpp"

#if defined(SPFFT_MPI) && (defined(SPFFT_CUDA) || defined(SPFFT_ROCM))
#include "mpi_util/mpi_communicator_handle.hpp"
#include "mpi_util/mpi_datatype_handle.hpp"
#include "mpi_util/mpi_request_handle.hpp"

namespace spfft {

template <typename T, typename U>
class TransposeMPICompactBufferedGPU : public Transpose {
  static_assert(IsFloatOrDouble<T>::value, "Type T must be float or double");

public:
  using ValueType = T;
  using ComplexType = std::complex<T>;
  using ComplexExchangeType = std::complex<U>;
  using ComplexGPUType = typename gpu::fft::ComplexType<T>::type;
  using ComplexExchangeGPUType = typename gpu::fft::ComplexType<U>::type;

  // spaceDomainDataGPU and freqDomainDataGPU must NOT overlap
  // spaceDomainDataGPU and spaceDomainBufferGPU must NOT overlap
  // freqDomainDataGPU and freqDomainBufferGPU must NOT overlap
  // spaceDomainBufferGPU and freqDomainBufferGPU must NOT overlap
  // spaceDomainBufferHost and freqDomainBufferHost must NOT overlap
  //
  // spaceDomainBufferGPU and freqDomainDataGPU MAY overlap
  // freqDomainBufferGPU and spaceDomainDataGPU MAY overlap
  TransposeMPICompactBufferedGPU(const std::shared_ptr<Parameters>& param,
                                 const std::vector<SpfftExchangeBackend>& exchBackends,
                                 MPICommunicatorHandle comm, GPUStreamHandle stream,
                                 HostArrayView1D<ComplexType> spaceDomainBufferHost,
                                 GPUArrayView3D<ComplexGPUType> spaceDomainDataGPU,
                                 GPUArrayView1D<ComplexGPUType> spaceDomainBufferGPU,
                                 HostArrayView1D<ComplexType> freqDomainBufferHost,
                                 GPUArrayView2D<ComplexGPUType> freqDomainDataGPU,
                                 GPUArrayView1D<ComplexGPUType> freqDomainBufferGPU);

  auto pack_backward() -> void override;
  auto exchange_backward_start(const bool nonBlockingExchange) -> void override;
  auto exchange_backward_finalize() -> void override;
  auto unpack_backward() -> void override;

  auto pack_forward() -> void override;
  auto exchange_forward_start(const bool nonBlockingExchange) -> void override;
  auto exchange_forward_finalize() -> void override;
  auto unpack_forward() -> void override;

  auto exchange_backend() -> SpfftExchangeBackend override { return exchBackend_; }

private:
  std::shared_ptr<Parameters> param_;
  SpfftExchangeBackend exchBackend_;
  MPIDatatypeHandle mpiTypeHandle_;
  MPICommunicatorHandle comm_;
  MPIRequestHandle mpiRequest_;

  HostArrayView1D<ComplexExchangeType> spaceDomainBufferHost_;
  HostArrayView1D<ComplexExchangeType> freqDomainBufferHost_;
  GPUArrayView3D<ComplexGPUType> spaceDomainDataGPU_;
  GPUArrayView2D<ComplexGPUType> freqDomainDataGPU_;
  GPUArrayView1D<ComplexExchangeGPUType> spaceDomainBufferGPU_;
  GPUArrayView1D<ComplexExchangeGPUType> freqDomainBufferGPU_;
  GPUStreamHandle stream_;

  GPUArray<int> numZSticksGPU_;
  GPUArray<int> numXYPlanesGPU_;
  GPUArray<int> xyPlaneOffsetsGPU_;
  GPUArray<int> indicesGPU_;

  class ExchangeImpl {
  public:
    virtual auto start(bool nonBlocking) -> void {}
    virtual auto finalize() -> void {}

    virtual ~ExchangeImpl() = default;
  };

  std::unique_ptr<ExchangeImpl> exchangeForward_;
  std::unique_ptr<ExchangeImpl> exchangeBackward_;

  // exchange impl
  class ExchangeIPC;
  class ExchangeMPI;
#ifdef SPFFT_NCCL
  class ExchangeNCCL;
#endif
};

}  // namespace spfft
#endif  // SPFFT_MPI
#endif
