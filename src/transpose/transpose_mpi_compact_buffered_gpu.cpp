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
#include <memory>
#include "gpu_util/gpu_runtime_api.hpp"
#include "spfft/config.h"
#if defined(SPFFT_MPI) && (defined(SPFFT_CUDA) || defined(SPFFT_ROCM))
#include <algorithm>
#include <cassert>
#include <complex>
#include <cstring>
#include <utility>
#include <vector>
#include <type_traits>
#include <tuple>
#include "memory/array_view_utility.hpp"
#include "memory/host_array_view.hpp"
#include "parameters/parameters.hpp"
#include "spfft/exceptions.hpp"
#include "transpose.hpp"
#include "util/common_types.hpp"
#include "util/omp_definitions.hpp"
#include "util/type_check.hpp"

#include "gpu_util/gpu_fft_api.hpp"
#include "gpu_util/gpu_transfer.hpp"
#include "mpi_util/mpi_check_status.hpp"
#include "mpi_util/mpi_communicator_handle.hpp"
#include "mpi_util/mpi_datatype_handle.hpp"
#include "mpi_util/mpi_match_elementary_type.hpp"
#include "transpose/gpu_kernels/compact_buffered_kernels.hpp"
#include "transpose/transpose_mpi_compact_buffered_gpu.hpp"

namespace spfft {

template <typename T, typename U>
class TransposeMPICompactBufferedGPU<T, U>::ExchangeIPC
    : public TransposeMPICompactBufferedGPU<T, U>::ExchangeImpl {
public:
  using ComplexExchangeGPUType =
      typename TransposeMPICompactBufferedGPU<T, U>::ComplexExchangeGPUType;

  ExchangeIPC(MPICommunicatorHandle comm, ComplexExchangeGPUType* target,
              ComplexExchangeGPUType* source, std::vector<int> localTargetDispls,
              const std::vector<int>& localSourceDispls, std::vector<int> count,
              GPUStreamHandle stream)
      : comm_(std::move(comm)),
        target_(target),
        localTargetDispls_(std::move(localTargetDispls)),
        remoteSourceDispls_(localSourceDispls.size()),
        count_(std::move(count)),
        stream_(std::move(stream)) {
    // mem handles
    gpu::IpcMemHandle localMemHandle;
    gpu::check_status(gpu::ipc_get_mem_handle(&localMemHandle, source));
    std::vector<gpu::IpcMemHandle> remoteMemHandles(comm_.size());
    mpi_check_status(MPI_Allgather(&localMemHandle, sizeof(decltype(localMemHandle)), MPI_BYTE,
                                   remoteMemHandles.data(), sizeof(decltype(localMemHandle)),
                                   MPI_BYTE, comm_.get()));
    for (SizeType r = 0; r < comm_.size(); ++r) {
      if (r == comm_.rank())
        sourceViews_.emplace_back(source);
      else
        sourceViews_.emplace_back(remoteMemHandles[r]);
    }

    // remote displs
    mpi_check_status(MPI_Alltoall(localSourceDispls.data(), 1, MPIMatchElementaryType<int>::get(),
                                  remoteSourceDispls_.data(), 1, MPIMatchElementaryType<int>::get(),
                                  comm_.get()));
  }

  auto start(bool nonBlocking) -> void override {
    gpu::stream_synchronize(stream_.get());
    mpi_check_status(MPI_Barrier(comm_.get()));

    // copy
    for (SizeType i = comm_.rank(); i < comm_.rank() + comm_.size(); ++i) {
      const auto r = i % comm_.size();
      if (count_[r]) {
        gpu::check_status(gpu::memcpy_async(target_ + localTargetDispls_[r],
                                            sourceViews_[r].get() + remoteSourceDispls_[r],
                                            count_[r] * sizeof(ComplexExchangeGPUType),
                                            gpu::flag::MemcpyDeviceToDevice, stream_.get()));
      }
    }
  }

  auto finalize() -> void override {
    gpu::stream_synchronize(stream_.get());
    mpi_check_status(MPI_Barrier(comm_.get()));
  }

private:
  MPICommunicatorHandle comm_;
  ComplexExchangeGPUType* target_;
  std::vector<GPUMemView<ComplexExchangeGPUType>> sourceViews_;
  std::vector<int> localTargetDispls_;
  std::vector<int> remoteSourceDispls_;
  std::vector<int> count_;
  GPUStreamHandle stream_;
};

#ifdef SPFFT_NCCL
template <typename T, typename U>
class TransposeMPICompactBufferedGPU<T, U>::ExchangeNCCL
    : public TransposeMPICompactBufferedGPU<T, U>::ExchangeImpl {
public:
  using ComplexExchangeGPUType =
      typename TransposeMPICompactBufferedGPU<T, U>::ComplexExchangeGPUType;

  ExchangeNCCL(MPICommunicatorHandle comm, ComplexExchangeGPUType* target,
              std::vector<int> targetCount, std::vector<int> targetDispls,
              ComplexExchangeGPUType* source, std::vector<int> sourceCount,
              std::vector<int> sourceDispls, GPUStreamHandle stream)
      : comm_(std::move(comm)),
        target_(target),
        source_(source),
        targetDispls_(std::move(targetDispls)),
        sourceDispls_(std::move(sourceDispls)),
        targetCount_(std::move(targetCount)),
        sourceCount_(std::move(sourceCount)),
        stream_(std::move(stream)) {}

  auto start(bool nonBlocking) -> void override {
    auto ncclType = ncclFloat64;
    if (std::is_same_v<ComplexExchangeGPUType, gpu::fft::ComplexType<float>::type>)
      ncclType = ncclFloat32;

    ncclGroupStart();
    for (SizeType r = 0; r < comm_.size(); ++r) {
      if (sourceCount_[r])
        nccl_check_status(ncclSend(source_ + sourceDispls_[r], 2 * sourceCount_[r], ncclType, r,
                                   comm_.get_nccl().get(), stream_.get()));
      if (targetCount_[r])
        nccl_check_status(ncclRecv(target_ + targetDispls_[r], 2 * targetCount_[r], ncclType, r,
                                   comm_.get_nccl().get(), stream_.get()));
    }
    ncclGroupEnd();
  }

  auto finalize() -> void override {}

private:
  MPICommunicatorHandle comm_;
  ComplexExchangeGPUType* target_;
  ComplexExchangeGPUType* source_;
  std::vector<int> targetDispls_;
  std::vector<int> sourceDispls_;
  std::vector<int> targetCount_;
  std::vector<int> sourceCount_;
  GPUStreamHandle stream_;
};
#endif

template <typename T, typename U>
class TransposeMPICompactBufferedGPU<T, U>::ExchangeMPI
    : public TransposeMPICompactBufferedGPU<T, U>::ExchangeImpl {
public:
  using ComplexExchangeGPUType =
      typename TransposeMPICompactBufferedGPU<T, U>::ComplexExchangeGPUType;

  ExchangeMPI(MPICommunicatorHandle comm, ComplexExchangeGPUType* target,
              std::vector<int> targetCount, std::vector<int> targetDispls,
              ComplexExchangeGPUType* source, std::vector<int> sourceCount,
              std::vector<int> sourceDispls, GPUStreamHandle stream)
      : comm_(std::move(comm)),
        target_(target),
        source_(source),
        targetDispls_(std::move(targetDispls)),
        sourceDispls_(std::move(sourceDispls)),
        targetCount_(std::move(targetCount)),
        sourceCount_(std::move(sourceCount)),
        stream_(std::move(stream)) {
    if (std::is_same_v<ComplexExchangeGPUType, gpu::fft::ComplexType<float>::type>)
      type_ = MPIDatatypeHandle::create_contiguous(2, MPIMatchElementaryType<float>::get());
    else
      type_ = MPIDatatypeHandle::create_contiguous(2, MPIMatchElementaryType<double>::get());
  }

  auto start(bool nonBlocking) -> void override {
    gpu::check_status(gpu::stream_synchronize(stream_.get()));

    if (nonBlocking) {
      mpi_check_status(MPI_Ialltoallv(source_, sourceCount_.data(), sourceDispls_.data(),
                                      type_.get(), target_, targetCount_.data(),
                                      targetDispls_.data(), type_.get(), comm_.get(),
                                      mpiRequest_.get_and_activate()));
    } else {
      mpi_check_status(MPI_Alltoallv(source_, sourceCount_.data(), sourceDispls_.data(),
                                     type_.get(), target_, targetCount_.data(),
                                     targetDispls_.data(), type_.get(), comm_.get()));
    }
  }

  auto finalize() -> void override { mpiRequest_.wait_if_active(); }

private:
  MPICommunicatorHandle comm_;
  ComplexExchangeGPUType* target_;
  ComplexExchangeGPUType* source_;
  std::vector<int> targetDispls_;
  std::vector<int> sourceDispls_;
  std::vector<int> targetCount_;
  std::vector<int> sourceCount_;
  GPUStreamHandle stream_;
  MPIRequestHandle mpiRequest_;
  MPIDatatypeHandle type_;
};

template <typename T, typename U>
TransposeMPICompactBufferedGPU<T, U>::TransposeMPICompactBufferedGPU(
    const std::shared_ptr<Parameters>& param, const std::vector<SpfftExchangeBackend>& exchBackends,
    MPICommunicatorHandle comm, GPUStreamHandle stream,
    HostArrayView1D<ComplexType> spaceDomainBufferHost,
    GPUArrayView3D<ComplexGPUType> spaceDomainDataGPU,
    GPUArrayView1D<ComplexGPUType> spaceDomainBufferGPU,
    HostArrayView1D<ComplexType> freqDomainBufferHost,
    GPUArrayView2D<ComplexGPUType> freqDomainDataGPU,
    GPUArrayView1D<ComplexGPUType> freqDomainBufferGPU)
    : param_(param),
      exchBackend_(SPFFT_EXCH_BACKEND_MPI_HOST),
      comm_(std::move(comm)),
      spaceDomainBufferHost_(create_new_type_1d_view<ComplexExchangeType>(
          spaceDomainBufferHost,
          param_->num_xy_planes(comm_.rank()) * param_->total_num_z_sticks())),
      freqDomainBufferHost_(create_new_type_1d_view<ComplexExchangeType>(
          freqDomainBufferHost,
          param_->total_num_xy_planes() * param_->num_z_sticks(comm_.rank()))),
      spaceDomainDataGPU_(spaceDomainDataGPU),
      freqDomainDataGPU_(freqDomainDataGPU),
      spaceDomainBufferGPU_(create_new_type_1d_view<ComplexExchangeGPUType>(
          spaceDomainBufferGPU,
          param_->total_num_z_sticks() * param_->num_xy_planes(comm_.rank()))),
      freqDomainBufferGPU_(create_new_type_1d_view<ComplexExchangeGPUType>(
          freqDomainBufferGPU, param_->num_z_sticks(comm_.rank()) * param_->total_num_xy_planes())),
      stream_(std::move(stream)) {
  assert(param_->dim_y() == spaceDomainDataGPU.dim_mid());
  assert(param_->dim_x_freq() == spaceDomainDataGPU.dim_inner());
  assert(param_->num_xy_planes(comm_.rank()) == spaceDomainDataGPU.dim_outer());
  assert(param_->dim_z() == freqDomainDataGPU.dim_inner());
  assert(param_->num_z_sticks(comm_.rank()) == freqDomainDataGPU.dim_outer());

  assert(spaceDomainBufferGPU.size() >=
         param_->total_num_z_sticks() * param_->num_xy_planes(comm_.rank()));
  assert(spaceDomainBufferHost.size() >=
         param_->total_num_z_sticks() * param_->num_xy_planes(comm_.rank()));
  assert(freqDomainBufferGPU.size() >=
         param_->total_num_xy_planes() * param_->num_z_sticks(comm_.rank()));
  assert(freqDomainBufferHost.size() >=
         param_->total_num_xy_planes() * param_->num_z_sticks(comm_.rank()));

  // assert(disjoint(spaceDomainDataGPU, freqDomainDataGPU));
  assert(disjoint(spaceDomainDataGPU, spaceDomainBufferGPU));
  assert(disjoint(freqDomainDataGPU, freqDomainBufferGPU));
  assert(disjoint(spaceDomainBufferHost, freqDomainBufferHost));
  if (exchBackend_ != SpfftExchangeBackend::SPFFT_EXCH_BACKEND_MPI_HOST) {
    assert(disjoint(spaceDomainBufferGPU, freqDomainBufferGPU));
  }

  // set exchange backend
  if (std::find(exchBackends.begin(), exchBackends.end(), SPFFT_EXCH_BACKEND_IPC) !=
      exchBackends.end()) {
    exchBackend_ = SPFFT_EXCH_BACKEND_IPC;
  } else if (std::find(exchBackends.begin(), exchBackends.end(), SPFFT_EXCH_BACKEND_NCCL) !=
             exchBackends.end()) {
    exchBackend_ = SPFFT_EXCH_BACKEND_NCCL;
  } else if (std::find(exchBackends.begin(), exchBackends.end(), SPFFT_EXCH_BACKEND_MPI_GPU) !=
             exchBackends.end()) {
    exchBackend_ = SPFFT_EXCH_BACKEND_MPI_GPU;
  } else if (std::find(exchBackends.begin(), exchBackends.end(), SPFFT_EXCH_BACKEND_MPI_HOST) !=
             exchBackends.end()) {
    exchBackend_ = SPFFT_EXCH_BACKEND_MPI_HOST;
  } else {
    throw InternalError();
  }

  // create underlying type
  mpiTypeHandle_ = MPIDatatypeHandle::create_contiguous(2, MPIMatchElementaryType<U>::get());

  // prepare mpi parameters
  std::vector<int> spaceDomainCount(comm_.size());
  std::vector<int> freqDomainCount(comm_.size());
  const SizeType numLocalZSticks = param_->num_z_sticks(comm_.rank());
  const SizeType numLocalXYPlanes = param_->num_xy_planes(comm_.rank());
  for (SizeType r = 0; r < (SizeType)comm_.size(); ++r) {
    freqDomainCount[r] = numLocalZSticks * param_->num_xy_planes(r);
    spaceDomainCount[r] = param_->num_z_sticks(r) * numLocalXYPlanes;
  }

  std::vector<int> spaceDomainDispls(comm_.size());
  std::vector<int> freqDomainDispls(comm_.size());
  int currentFreqDomainDispls = 0;
  int currentSpaceDomainDispls = 0;
  for (SizeType r = 0; r < (SizeType)comm_.size(); ++r) {
    assert(currentSpaceDomainDispls + spaceDomainCount[r] <=
           static_cast<int>(spaceDomainBufferHost.size()));
    assert(currentFreqDomainDispls + freqDomainCount[r] <=
           static_cast<int>(freqDomainBufferHost.size()));
    spaceDomainDispls[r] = currentSpaceDomainDispls;
    freqDomainDispls[r] = currentFreqDomainDispls;
    currentSpaceDomainDispls += spaceDomainCount[r];
    currentFreqDomainDispls += freqDomainCount[r];
  }

  // copy relevant parameters to gpu
  std::vector<int> numZSticksHost(comm_.size());
  std::vector<int> numXYPlanesHost(comm_.size());
  std::vector<int> xyPlaneOffsetsHost(comm_.size());
  std::vector<int> indicesHost(comm_.size() * param_->max_num_z_sticks());
  for (SizeType r = 0; r < comm_.size(); ++r) {
    numZSticksHost[r] = static_cast<int>(param_->num_z_sticks(r));
    numXYPlanesHost[r] = static_cast<int>(param_->num_xy_planes(r));
    xyPlaneOffsetsHost[r] = static_cast<int>(param_->xy_plane_offset(r));
    const auto zStickXYIndices = param_->z_stick_xy_indices(r);
    for (SizeType i = 0; i < zStickXYIndices.size(); ++i) {
      // transpose stick index
      const int xyIndex = zStickXYIndices(i);
      const int x = xyIndex / param_->dim_y();
      const int y = xyIndex - x * param_->dim_y();
      indicesHost[r * param_->max_num_z_sticks() + i] = y * param_->dim_x_freq() + x;
    }
  }
  numZSticksGPU_ = GPUArray<int>(numZSticksHost.size());
  numXYPlanesGPU_ = GPUArray<int>(numXYPlanesHost.size());
  xyPlaneOffsetsGPU_ = GPUArray<int>(xyPlaneOffsetsHost.size());
  indicesGPU_ = GPUArray<int>(indicesHost.size());

  copy_to_gpu(numZSticksHost, numZSticksGPU_);
  copy_to_gpu(numXYPlanesHost, numXYPlanesGPU_);
  copy_to_gpu(xyPlaneOffsetsHost, xyPlaneOffsetsGPU_);
  copy_to_gpu(indicesHost, indicesGPU_);

  if (exchBackend_ == SPFFT_EXCH_BACKEND_NCCL) {
#ifdef SPFFT_NCCL
    if (!comm_.init_nccl()) throw InternalError();
    exchangeBackward_ = std::make_unique<TransposeMPICompactBufferedGPU<T, U>::ExchangeNCCL>(
        comm_, spaceDomainBufferGPU_.data(), spaceDomainCount, spaceDomainDispls,
        freqDomainBufferGPU_.data(), freqDomainCount, freqDomainDispls, stream_);
    exchangeForward_ = std::make_unique<TransposeMPICompactBufferedGPU<T, U>::ExchangeNCCL>(
        comm_, freqDomainBufferGPU_.data(), freqDomainCount, freqDomainDispls,
        spaceDomainBufferGPU_.data(), spaceDomainCount, spaceDomainDispls, stream_);
#else
    throw InternalError();
#endif
  }

  if (exchBackend_ == SPFFT_EXCH_BACKEND_IPC) {
    exchangeBackward_ = std::make_unique<TransposeMPICompactBufferedGPU<T, U>::ExchangeIPC>(
        comm_, spaceDomainBufferGPU_.data(), freqDomainBufferGPU_.data(), spaceDomainDispls,
        freqDomainDispls, spaceDomainCount, stream_);
    exchangeForward_ = std::make_unique<TransposeMPICompactBufferedGPU<T, U>::ExchangeIPC>(
        comm_, freqDomainBufferGPU_.data(), spaceDomainBufferGPU_.data(), freqDomainDispls,
        spaceDomainDispls, freqDomainCount, stream_);
  }

  if (exchBackend_ == SPFFT_EXCH_BACKEND_MPI_HOST || exchBackend_ == SPFFT_EXCH_BACKEND_MPI_GPU) {
    ComplexExchangeGPUType* freqPtr =
        reinterpret_cast<ComplexExchangeGPUType*>(freqDomainBufferHost_.data());
    ComplexExchangeGPUType* spacePtr =
        reinterpret_cast<ComplexExchangeGPUType*>(spaceDomainBufferHost_.data());
    if (exchBackend_ != SpfftExchangeBackend::SPFFT_EXCH_BACKEND_MPI_HOST) {
      freqPtr = freqDomainBufferGPU_.data();
      spacePtr = spaceDomainBufferGPU_.data();
    }

    exchangeBackward_ = std::make_unique<TransposeMPICompactBufferedGPU<T, U>::ExchangeMPI>(
        comm_, spacePtr, spaceDomainCount, spaceDomainDispls, freqPtr, freqDomainCount,
        freqDomainDispls, stream_);
    exchangeForward_ = std::make_unique<TransposeMPICompactBufferedGPU<T, U>::ExchangeMPI>(
        comm_, freqPtr, freqDomainCount, freqDomainDispls, spacePtr, spaceDomainCount,
        spaceDomainDispls, stream_);
  }
}

template <typename T, typename U>
auto TransposeMPICompactBufferedGPU<T, U>::pack_backward() -> void {
  if (freqDomainDataGPU_.size() > 0 && freqDomainBufferGPU_.size() > 0) {
    compact_buffered_pack_backward(stream_.get(), param_->max_num_xy_planes(),
                                   create_1d_view(numXYPlanesGPU_, 0, numXYPlanesGPU_.size()),
                                   create_1d_view(xyPlaneOffsetsGPU_, 0, xyPlaneOffsetsGPU_.size()),
                                   freqDomainDataGPU_, freqDomainBufferGPU_);
    if (exchBackend_ == SpfftExchangeBackend::SPFFT_EXCH_BACKEND_MPI_HOST)
      copy_from_gpu_async(stream_, freqDomainBufferGPU_, freqDomainBufferHost_);
  }
}

template <typename T, typename U>
auto TransposeMPICompactBufferedGPU<T, U>::unpack_backward() -> void {
  if (spaceDomainDataGPU_.size() > 0) {
    gpu::check_status(gpu::memset_async(
        static_cast<void*>(spaceDomainDataGPU_.data()), 0,
        spaceDomainDataGPU_.size() * sizeof(typename decltype(spaceDomainDataGPU_)::ValueType),
        stream_.get()));
    if (spaceDomainBufferGPU_.size() > 0) {
      if (exchBackend_ == SpfftExchangeBackend::SPFFT_EXCH_BACKEND_MPI_HOST)
        copy_to_gpu_async(stream_, spaceDomainBufferHost_, spaceDomainBufferGPU_);

      compact_buffered_unpack_backward(stream_.get(), param_->max_num_z_sticks(),
                                       create_1d_view(numZSticksGPU_, 0, numZSticksGPU_.size()),
                                       create_1d_view(indicesGPU_, 0, indicesGPU_.size()),
                                       spaceDomainBufferGPU_, spaceDomainDataGPU_);
    }
  }
}

template <typename T, typename U>
auto TransposeMPICompactBufferedGPU<T, U>::exchange_backward_start(const bool nonBlockingExchange)
    -> void {
  assert(omp_get_thread_num() == 0);  // only must thread must be allowed to enter
  exchangeBackward_->start(nonBlockingExchange);
}

template <typename T, typename U>
auto TransposeMPICompactBufferedGPU<T, U>::exchange_backward_finalize() -> void {
  exchangeBackward_->finalize();
}

template <typename T, typename U>
auto TransposeMPICompactBufferedGPU<T, U>::pack_forward() -> void {
  if (spaceDomainDataGPU_.size() > 0 && spaceDomainBufferGPU_.size() > 0) {
    compact_buffered_pack_forward(stream_.get(), param_->max_num_z_sticks(),
                                  create_1d_view(numZSticksGPU_, 0, numZSticksGPU_.size()),
                                  create_1d_view(indicesGPU_, 0, indicesGPU_.size()),
                                  spaceDomainDataGPU_, spaceDomainBufferGPU_);

    if (exchBackend_ == SpfftExchangeBackend::SPFFT_EXCH_BACKEND_MPI_HOST)
      copy_from_gpu_async(stream_, spaceDomainBufferGPU_, spaceDomainBufferHost_);
  }
}

template <typename T, typename U>
auto TransposeMPICompactBufferedGPU<T, U>::unpack_forward() -> void {
  if (freqDomainDataGPU_.size() > 0 && freqDomainBufferGPU_.size() > 0) {
    if (exchBackend_ == SpfftExchangeBackend::SPFFT_EXCH_BACKEND_MPI_HOST)
      copy_to_gpu_async(stream_, freqDomainBufferHost_, freqDomainBufferGPU_);

    compact_buffered_unpack_forward(
        stream_.get(), param_->max_num_xy_planes(),
        create_1d_view(numXYPlanesGPU_, 0, numXYPlanesGPU_.size()),
        create_1d_view(xyPlaneOffsetsGPU_, 0, xyPlaneOffsetsGPU_.size()), freqDomainBufferGPU_,
        freqDomainDataGPU_);
  }
}

template <typename T, typename U>
auto TransposeMPICompactBufferedGPU<T, U>::exchange_forward_start(const bool nonBlockingExchange)
    -> void {
  assert(omp_get_thread_num() == 0);  // only must thread must be allowed to enter
  exchangeForward_->start(nonBlockingExchange);
}

template <typename T, typename U>
auto TransposeMPICompactBufferedGPU<T, U>::exchange_forward_finalize() -> void {
  exchangeForward_->finalize();
}

// Instantiate class for float and double
#ifdef SPFFT_SINGLE_PRECISION
template class TransposeMPICompactBufferedGPU<float, float>;
#endif
template class TransposeMPICompactBufferedGPU<double, double>;
template class TransposeMPICompactBufferedGPU<double, float>;
}  // namespace spfft
#endif  // SPFFT_MPI
