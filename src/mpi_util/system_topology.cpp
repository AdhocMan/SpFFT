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

#include "mpi_util/system_topology.hpp"

#ifdef SPFFT_MPI

#include <mpi.h>
#include <unistd.h>

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "mpi_util/mpi_check_status.hpp"
#include "spfft/exceptions.hpp"
#include "util/common_types.hpp"

#if (defined(SPFFT_CUDA) || defined(SPFFT_ROCM))
#include "gpu_util/gpu_runtime_api.hpp"
#endif

namespace spfft {

namespace {
struct HostDeviceHashses {
  unsigned long long host = 0;
  unsigned long long device = 0;
};
}  // namespace

SystemTopology::SystemTopology(const MPI_Comm& comm, SpfftProcessingUnitType pu)
    : numNodes(1), numDevices(0) {
  int commRank, commSize;
  mpi_check_status(MPI_Comm_size(comm, &commSize));
  mpi_check_status(MPI_Comm_rank(comm, &commRank));

  HostDeviceHashses localHash;

  // generate unique host machine hash
  std::string hostName;
  hostName.resize(1024);
  std::ignore = gethostname(hostName.data(), hostName.size());
  localHash.host = std::hash<std::string>{}(hostName);

  std::ifstream bootIdFile("/proc/sys/kernel/random/boot_id");
  if (bootIdFile.is_open()) {
    std::stringstream fileStream;
    fileStream << bootIdFile.rdbuf();
    std::string bootString = fileStream.str();
    localHash.host ^= std::hash<std::string>{}(bootString);
  }

#if (defined(SPFFT_CUDA) || defined(SPFFT_ROCM))
  if (pu & SpfftProcessingUnitType::SPFFT_PU_GPU) {
    // generate GPU hash
    std::string gpuPCIId;
    gpuPCIId.resize(13);
    int deviceId = 0;
    gpu::check_status(gpu::get_device(&deviceId));
    gpu::check_status(gpu::device_get_pcibusid(gpuPCIId.data(), gpuPCIId.size(), deviceId));
    localHash.device = std::hash<std::string>{}(gpuPCIId);
    this->numDevices += 1;
  }
#endif

  // gather hashes
  std::vector<HostDeviceHashses> hashes(commSize);
  mpi_check_status(MPI_Allgather(&localHash, 2, MPI_UNSIGNED_LONG_LONG, hashes.data(), 2,
                                 MPI_UNSIGNED_LONG_LONG, comm));

  // sort to compare
  std::sort(hashes.begin(), hashes.end(),
            [](const HostDeviceHashses& a, const HostDeviceHashses& b) {
              if (a.host != b.host) return a.host < b.host;
              return a.device < b.device;
            });

  // count other nodes / devices. Hash of 0 for device indicates no device.
  for (SizeType r = 0; r < commSize - 1; ++r) {
    if (hashes[r].host != hashes[r + 1].host) {
      // different node
      this->numNodes += 1;
      if (hashes[r].device) this->numDevices += 1;
    } else if (hashes[r].device != hashes[r + 1].device && hashes[r].device) {
      // same node
      this->numDevices += 1;
    }
  }
}

}  // namespace spfft

#endif
