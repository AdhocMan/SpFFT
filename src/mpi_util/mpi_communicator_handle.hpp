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
#ifndef SPFFT_MPI_COMMUNICATOR_HANDLE_HPP
#define SPFFT_MPI_COMMUNICATOR_HANDLE_HPP

#include <mpi.h>
#include <unistd.h>

#include <cassert>
#include <cstddef>
#include <fstream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "mpi_util/mpi_check_status.hpp"
#include "spfft/config.h"
#include "spfft/exceptions.hpp"
#include "util/common_types.hpp"

#ifdef SPFFT_NCCL
#include "gpu_util/nccl_comm_handle.hpp"
#if defined(SPFFT_CUDA) || defined(SPFFT_ROCM)
#include "gpu_util/gpu_runtime_api.hpp"
#endif
#endif


namespace spfft {

// MPI Communicator, which creates a duplicate at construction time.
// Copies of the object share the same communicator, which is reference counted.
class MPICommunicatorHandle {
public:
  MPICommunicatorHandle() : comm_(new MPI_Comm(MPI_COMM_SELF)), size_(1), rank_(0) {}

  MPICommunicatorHandle(const MPI_Comm& comm) {
    // create copy of communicator
    MPI_Comm newComm;
    mpi_check_status(MPI_Comm_dup(comm, &newComm));

    comm_ = std::shared_ptr<MPI_Comm>(new MPI_Comm(newComm), [](MPI_Comm* ptr) {
      int finalized = 0;
      MPI_Finalized(&finalized);
      if (!finalized) {
        MPI_Comm_free(ptr);
      }
      delete ptr;
    });

    int sizeInt, rankInt;
    mpi_check_status(MPI_Comm_size(*comm_, &sizeInt));
    mpi_check_status(MPI_Comm_rank(*comm_, &rankInt));

    if (sizeInt < 1 || rankInt < 0) {
      throw MPIError();
    }
    rank_ = static_cast<SizeType>(rankInt);
    size_ = static_cast<SizeType>(sizeInt);
  }

  inline auto get() const -> const MPI_Comm& { return *comm_; }

  inline auto size() const noexcept -> SizeType { return size_; }

  inline auto rank() const noexcept -> SizeType { return rank_; }

  inline auto has_nccl() const noexcept -> bool {
#ifdef SPFFT_NCCL
    return ncclComm_.has_value();
#else
    return false;
#endif
  }

#ifdef SPFFT_NCCL
  inline auto get_nccl() -> const NCCLCommHandle& {
    return ncclComm_.value();
  }

  inline auto init_nccl() -> void {
    // check if multiple MPI ranks per GPU

    // generate unique host machine hash
    std::string hostName;
    hostName.resize(1024);
    std::ignore = gethostname(hostName.data(), hostName.size());
    auto hostHash = std::hash<std::string>{}(hostName);

    std::ifstream bootIdFile("/proc/sys/kernel/random/boot_id");
    if (bootIdFile.is_open()) {
      std::stringstream fileStream;
      fileStream << bootIdFile.rdbuf();
      std::string bootString = fileStream.str();
      hostHash ^= std::hash<std::string>{}(bootString);
    }

    // generate GPU hash
    std::string gpuPCIId;
    gpuPCIId.resize(13);
    int deviceId = 0;
    gpu::check_status(gpu::get_device(&deviceId));
    gpu::check_status(gpu::device_get_pcibusid(gpuPCIId.data(), gpuPCIId.size(), deviceId));
    auto deviceHash = std::hash<std::string>{}(gpuPCIId);

    // compare hashes
    std::vector<decltype(hostHash)> hostHashes(this->size());
    mpi_check_status(MPI_Allgather(&hostHash, sizeof(decltype(hostHash)), MPI_BYTE,
                                   hostHashes.data(), sizeof(decltype(hostHash)), MPI_BYTE,
                                   this->get()));
    std::vector<decltype(deviceHash)> deviceHashes(this->size());
    mpi_check_status(MPI_Allgather(&deviceHash, sizeof(decltype(deviceHash)), MPI_BYTE,
                                   deviceHashes.data(), sizeof(decltype(deviceHash)), MPI_BYTE,
                                   this->get()));
    bool oversubsribed = false;
    for (SizeType r = 0; r < this->size(); ++r) {
      if (r == this->rank()) continue;
      if (deviceHashes[r] == deviceHash && hostHashes[r] == hostHash) oversubsribed = true;
    }
    mpi_check_status(
        MPI_Allreduce(MPI_IN_PLACE, &oversubsribed, 1, MPI_C_BOOL, MPI_LOR, this->get()));

    // Try to initialize NCCL. Fails, if there are multiple MPI ranks per GPU.
    if (!oversubsribed) {
      try {
        ncclComm_t comm;

        ncclUniqueId id;
        if (this->rank() == 0) nccl_check_status(ncclGetUniqueId(&id));
        MPI_Bcast(&id, sizeof(decltype(id)), MPI_BYTE, 0, this->get());

        nccl_check_status(ncclGroupStart());
        nccl_check_status(ncclCommInitRank(&comm, this->size(), id, this->rank()));
        nccl_check_status(ncclGroupEnd());
        ncclComm_ = NCCLCommHandle(comm);
      } catch (...) {
      }
    }
  }
#endif

private:
  std::shared_ptr<MPI_Comm> comm_ = nullptr;
  SizeType size_ = 1;
  SizeType rank_ = 0;

#ifdef SPFFT_NCCL
  std::optional<NCCLCommHandle> ncclComm_;
#endif
};

}  // namespace spfft

#endif
