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
TransposeMPICompactBufferedGPU<T, U>::TransposeMPICompactBufferedGPU(
    const std::shared_ptr<Parameters>& param, SpfftExchangeBackend exchBackend,
    MPICommunicatorHandle comm, GPUStreamHandle stream,
    HostArrayView1D<ComplexType> spaceDomainBufferHost,
    GPUArrayView3D<ComplexGPUType> spaceDomainDataGPU,
    GPUArrayView1D<ComplexGPUType> spaceDomainBufferGPU,
    HostArrayView1D<ComplexType> freqDomainBufferHost,
    GPUArrayView2D<ComplexGPUType> freqDomainDataGPU,
    GPUArrayView1D<ComplexGPUType> freqDomainBufferGPU)
    : param_(param),
      exchBackend_(exchBackend),
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

#ifndef SPFFT_NCCL
  assert(exchBackend_ != SPFFT_EXCH_BACKEND_NCCL);
#endif
#ifndef SPFFT_GPU_DIRECT
  assert(exchBackend_ != SPFFT_EXCH_BACKEND_MPI_GPU);
#endif

  // create underlying type
  mpiTypeHandle_ = MPIDatatypeHandle::create_contiguous(2, MPIMatchElementaryType<U>::get());

  // prepare mpi parameters
  spaceDomainCount_.resize(comm_.size());
  freqDomainCount_.resize(comm_.size());
  const SizeType numLocalZSticks = param_->num_z_sticks(comm_.rank());
  const SizeType numLocalXYPlanes = param_->num_xy_planes(comm_.rank());
  for (SizeType r = 0; r < (SizeType)comm_.size(); ++r) {
    freqDomainCount_[r] = numLocalZSticks * param_->num_xy_planes(r);
    spaceDomainCount_[r] = param_->num_z_sticks(r) * numLocalXYPlanes;
  }

  spaceDomainDispls_.resize(comm_.size());
  freqDomainDispls_.resize(comm_.size());
  int currentFreqDomainDispls = 0;
  int currentSpaceDomainDispls = 0;
  for (SizeType r = 0; r < (SizeType)comm_.size(); ++r) {
    assert(currentSpaceDomainDispls + spaceDomainCount_[r] <=
           static_cast<int>(spaceDomainBufferHost.size()));
    assert(currentFreqDomainDispls + freqDomainCount_[r] <=
           static_cast<int>(freqDomainBufferHost.size()));
    spaceDomainDispls_[r] = currentSpaceDomainDispls;
    freqDomainDispls_[r] = currentFreqDomainDispls;
    currentSpaceDomainDispls += spaceDomainCount_[r];
    currentFreqDomainDispls += freqDomainCount_[r];
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

  try {
    // events
    remoteEvents_.clear();
    GPUEventHandle localEvent(false, true);
    gpu::IpcEventHandle ipcEvent;
    gpu::check_status(gpu::ipc_get_event_handle(&ipcEvent, localEvent.get()));

    std::vector<gpu::IpcEventHandle> remoteIpcEventHandles(comm_.size());
    mpi_check_status(MPI_Allgather(&ipcEvent, sizeof(decltype(ipcEvent)), MPI_BYTE,
                                   remoteIpcEventHandles.data(), sizeof(decltype(ipcEvent)),
                                   MPI_BYTE, comm_.get()));

    for(SizeType r = 0; r < comm_.size(); ++r) {
      if(r== comm_.rank()) {
        remoteEvents_.emplace_back(std::move(localEvent));
      } else {
        remoteEvents_.emplace_back(remoteIpcEventHandles[r]);
      }
    }

    // mem handles
    gpu::IpcMemHandle localFreqHandle;
    gpu::check_status(gpu::ipc_get_mem_handle(&localFreqHandle, freqDomainBufferGPU_.data()));
    std::vector<gpu::IpcMemHandle> remoteFreqMemHandles(comm_.size());
    mpi_check_status(MPI_Allgather(&localFreqHandle, sizeof(decltype(localFreqHandle)), MPI_BYTE,
                                   remoteFreqMemHandles.data(), sizeof(decltype(localFreqHandle)),
                                   MPI_BYTE, comm_.get()));
    for (SizeType r = 0; r < comm_.size(); ++r) {
      if (r == comm_.rank())
        remoteFreqDomainGPU_.emplace_back(freqDomainBufferGPU_.data());
      else
        remoteFreqDomainGPU_.emplace_back(remoteFreqMemHandles[r]);
    }

    gpu::IpcMemHandle localSpaceHandle;
    gpu::check_status(gpu::ipc_get_mem_handle(&localSpaceHandle, spaceDomainBufferGPU_.data()));
    std::vector<gpu::IpcMemHandle> remoteSpaceMemHandles(comm_.size());
    mpi_check_status(MPI_Allgather(&localSpaceHandle, sizeof(decltype(localSpaceHandle)), MPI_BYTE,
                                   remoteSpaceMemHandles.data(), sizeof(decltype(localSpaceHandle)),
                                   MPI_BYTE, comm_.get()));
    for(SizeType r = 0; r < comm_.size(); ++r) {
      if (r == comm_.rank())
        remoteSpaceDomainGPU_.emplace_back(spaceDomainBufferGPU_.data());
      else
        remoteSpaceDomainGPU_.emplace_back(remoteSpaceMemHandles[r]);
    }

    remoteSpaceDomainDispls_.resize(comm_.size());
    remoteFreqDomainDispls_.resize(comm_.size());

    mpi_check_status(MPI_Alltoall(freqDomainDispls_.data(), 1, MPIMatchElementaryType<int>::get(),
                                  remoteFreqDomainDispls_.data(), 1,
                                  MPIMatchElementaryType<int>::get(), comm_.get()));
    mpi_check_status(MPI_Alltoall(spaceDomainDispls_.data(), 1, MPIMatchElementaryType<int>::get(),
                                  remoteSpaceDomainDispls_.data(), 1,
                                  MPIMatchElementaryType<int>::get(), comm_.get()));

    exchBackend_ = SPFFT_EXCH_BACKEND_IPC;

  } catch (...) {
    remoteEvents_.clear();
    remoteFreqDomainGPU_.clear();
    remoteSpaceDomainGPU_.clear();
    printf("IPC ERROR!\n");
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

  if(exchBackend_ == SpfftExchangeBackend::SPFFT_EXCH_BACKEND_IPC) {
    // barrier
    if(comm_.rank() == 0){
      for(SizeType r = 1; r < comm_.size(); ++r){
        gpu::check_status(gpu::stream_wait_event(stream_.get(), remoteEvents_[r].get(), 0));
        gpu::check_status(gpu::event_record(remoteEvents_[0].get(), stream_.get()));
      }
    } else {
      gpu::check_status(gpu::event_record(remoteEvents_[comm_.rank()].get(), stream_.get()));
      gpu::check_status(gpu::stream_wait_event(stream_.get(), remoteEvents_[0].get(), 0));
    }

    // gpu::device_synchronize();
    // mpi_check_status(MPI_Barrier(comm_.get()));

    // copy
    for (SizeType i = comm_.rank(); i < comm_.rank() + comm_.size(); ++i) {
      const auto r = i % comm_.size();

      if (spaceDomainCount_[r]) {
        gpu::check_status(
            gpu::memcpy_async(spaceDomainBufferGPU_.data() + spaceDomainDispls_[r],
                              remoteFreqDomainGPU_[r].get() + remoteFreqDomainDispls_[r],
                              spaceDomainCount_[r] * sizeof(ComplexExchangeGPUType),
                              gpu::flag::MemcpyDeviceToDevice, stream_.get()));
      }
    }

    // gpu::device_synchronize();
    // mpi_check_status(MPI_Barrier(comm_.get()));

    // barrier
    if (comm_.rank() == 0) {
      for (SizeType r = 1; r < comm_.size(); ++r) {
        gpu::check_status(gpu::stream_wait_event(stream_.get(), remoteEvents_[r].get(), 0));
        gpu::check_status(gpu::event_record(remoteEvents_[0].get(), stream_.get()));
      }
    } else {
      gpu::check_status(gpu::event_record(remoteEvents_[comm_.rank()].get(), stream_.get()));
      gpu::check_status(gpu::stream_wait_event(stream_.get(), remoteEvents_[0].get(), 0));
    }


    // exit
    return;
  }

#ifdef SPFFT_NCCL
  if (exchBackend_ == SpfftExchangeBackend::SPFFT_EXCH_BACKEND_NCCL) {
    auto ncclType = ncclFloat64;
    if (std::is_same_v<U, float>) ncclType = ncclFloat32;

    ncclGroupStart();
    for (SizeType r = 0; r < comm_.size(); ++r) {
      if(freqDomainCount_[r])
        nccl_check_status(ncclSend(freqDomainBufferGPU_.data() + freqDomainDispls_[r],
                                   2 * freqDomainCount_[r], ncclType, r, comm_.get_nccl().get(),
                                   stream_.get()));
      if(spaceDomainCount_[r])
        nccl_check_status(ncclRecv(spaceDomainBufferGPU_.data() + spaceDomainDispls_[r],
                                   2 * spaceDomainCount_[r], ncclType, r, comm_.get_nccl().get(),
                                   stream_.get()));
    }
    ncclGroupEnd();

    // exit
    return;
  }
#endif

  void* sendBufferPtr = freqDomainBufferHost_.data();
  void* recvBufferPtr = spaceDomainBufferHost_.data();
  if (exchBackend_ != SpfftExchangeBackend::SPFFT_EXCH_BACKEND_MPI_HOST) {
    sendBufferPtr = freqDomainBufferGPU_.data();
    recvBufferPtr = spaceDomainBufferGPU_.data();
  }

  gpu::check_status(gpu::stream_synchronize(stream_.get()));

  if (nonBlockingExchange) {
    mpi_check_status(MPI_Ialltoallv(
        sendBufferPtr, freqDomainCount_.data(), freqDomainDispls_.data(), mpiTypeHandle_.get(),
        recvBufferPtr, spaceDomainCount_.data(), spaceDomainDispls_.data(), mpiTypeHandle_.get(),
        comm_.get(), mpiRequest_.get_and_activate()));
  } else {
    mpi_check_status(MPI_Alltoallv(sendBufferPtr, freqDomainCount_.data(), freqDomainDispls_.data(),
                                   mpiTypeHandle_.get(), recvBufferPtr, spaceDomainCount_.data(),
                                   spaceDomainDispls_.data(), mpiTypeHandle_.get(), comm_.get()));
  }
}

template <typename T, typename U>
auto TransposeMPICompactBufferedGPU<T, U>::exchange_backward_finalize() -> void {
  mpiRequest_.wait_if_active();
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

#ifdef SPFFT_NCCL
  if (exchBackend_ == SpfftExchangeBackend::SPFFT_EXCH_BACKEND_NCCL) {
    auto ncclType = ncclFloat64;
    if (std::is_same_v<U, float>) ncclType = ncclFloat32;

    ncclGroupStart();
    for (SizeType r = 0; r < comm_.size(); ++r) {
      // if(r == comm_.rank()) continue;
      if (spaceDomainCount_[r])
        nccl_check_status(ncclSend(spaceDomainBufferGPU_.data() + spaceDomainDispls_[r],
                                   2 * spaceDomainCount_[r], ncclType, r, comm_.get_nccl().get(),
                                   stream_.get()));
      if (freqDomainCount_[r])
        nccl_check_status(ncclRecv(freqDomainBufferGPU_.data() + freqDomainDispls_[r],
                                   2 * freqDomainCount_[r], ncclType, r, comm_.get_nccl().get(),
                                   stream_.get()));
    }
    ncclGroupEnd();

    // exit
    return;
  }
#endif


  gpu::check_status(gpu::stream_synchronize(stream_.get()));

  void* sendBufferPtr = spaceDomainBufferHost_.data();
  void* recvBufferPtr = freqDomainBufferHost_.data();

  if (exchBackend_ != SpfftExchangeBackend::SPFFT_EXCH_BACKEND_MPI_HOST) {
    sendBufferPtr = spaceDomainBufferGPU_.data();
    recvBufferPtr = freqDomainBufferGPU_.data();
  }

  if (nonBlockingExchange) {
    // start non-blocking exchange
    mpi_check_status(MPI_Ialltoallv(
        sendBufferPtr, spaceDomainCount_.data(), spaceDomainDispls_.data(), mpiTypeHandle_.get(),
        recvBufferPtr, freqDomainCount_.data(), freqDomainDispls_.data(), mpiTypeHandle_.get(),
        comm_.get(), mpiRequest_.get_and_activate()));
  } else {
    // blocking exchange
    mpi_check_status(MPI_Alltoallv(sendBufferPtr, spaceDomainCount_.data(),
                                   spaceDomainDispls_.data(), mpiTypeHandle_.get(), recvBufferPtr,
                                   freqDomainCount_.data(), freqDomainDispls_.data(),
                                   mpiTypeHandle_.get(), comm_.get()));
  }
}

template <typename T, typename U>
auto TransposeMPICompactBufferedGPU<T, U>::exchange_forward_finalize() -> void {
  mpiRequest_.wait_if_active();
}

// Instantiate class for float and double
#ifdef SPFFT_SINGLE_PRECISION
template class TransposeMPICompactBufferedGPU<float, float>;
#endif
template class TransposeMPICompactBufferedGPU<double, double>;
template class TransposeMPICompactBufferedGPU<double, float>;
}  // namespace spfft
#endif  // SPFFT_MPI
