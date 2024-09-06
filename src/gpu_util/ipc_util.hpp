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
#ifndef SPFFT_IPC_UTIL_HPP
#define SPFFT_IPC_UTIL_HPP

#include "spfft/config.h"
#include "spfft/exceptions.hpp"

#if (defined(SPFFT_CUDA) || defined(SPFFT_ROCM)) && defined(SPFFT_MPI)
#include <tuple>
#include <vector>

#include "gpu_util/gpu_runtime_api.hpp"
#include "memory/gpu_array.hpp"
#include "spfft/exceptions.hpp"
#include "mpi_util/mpi_communicator_handle.hpp"

namespace spfft {

  inline auto gpu_ipc_available(const MPICommunicatorHandle& comm) -> bool {
    GPUArray<double> localArray(1);

    bool available = true;
    try {
      gpu::IpcMemHandle localMemHandle;
      gpu::check_status(gpu::ipc_get_mem_handle(&localMemHandle, localArray.data()));
      std::vector<gpu::IpcMemHandle> remoteMemHandles(comm.size());
      mpi_check_status(MPI_Allgather(&localMemHandle, sizeof(decltype(localMemHandle)), MPI_BYTE,
                                     remoteMemHandles.data(), sizeof(decltype(localMemHandle)),
                                     MPI_BYTE, comm.get()));

      for (SizeType r = 0; r < comm.size(); ++r) {
        if (r != comm.rank()) {
          void* remotePtr;
          auto statusOpen = gpu::ipc_open_mem_handle(&remotePtr, remoteMemHandles[r],
                                                     gpu::flag::IpcMemLazyEnablePeerAccess);
          auto statusClose = gpu::ipc_close_mem_handle(remotePtr);
          if (statusOpen != gpu::status::Success || statusClose != gpu::status::Success) {
            // clear error
            std::ignore = gpu::get_last_error();
            available = false;
            break;
          }
        }
      }

    } catch (...) {
      // clear error
      std::ignore = gpu::get_last_error();
      available = false;
    }

    mpi_check_status(MPI_Allreduce(MPI_IN_PLACE, &available, 1, MPI_C_BOOL, MPI_LAND, comm.get()));

    return available;
  }

}  // namespace spfft

#endif
#endif
