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
#ifndef SPFFT_NCCL_COMM_HANDLE_HPP
#define SPFFT_NCCL_COMM_HANDLE_HPP

#include "spfft/config.h"
#if defined(SPFFT_NCCL) | defined(SPFFT_MPI)
#include <memory>
#include <tuple>
#include <nccl.h>
#include "spfft/exceptions.hpp"
#include "gpu_util/nccl_check_status.hpp"

namespace spfft {
class NCCLCommHandle {
public:
  // explicit NCCLCommHandle()  {
  //   ncclComm_t comm;

  //   ncclUniqueId id;
  //   nccl_check_status(ncclGetUniqueId(&id));
  //   nccl_check_status(ncclGroupStart());
  //   nccl_check_status(ncclCommInitRank(&comm, 1, id, 0));
  //   nccl_check_status(ncclGroupEnd());

  //   comm_ = std::shared_ptr<ncclComm_t>(new ncclComm_t(comm), [](ncclComm_t* ptr) {
  //     std::ignore = ncclCommDestroy(*ptr);
  //     delete ptr;
  //   });
  // };

  explicit NCCLCommHandle(ncclComm_t comm) {
    comm_ = std::shared_ptr<ncclComm_t>(new ncclComm_t(comm), [](ncclComm_t* ptr) {
      std::ignore = ncclCommDestroy(*ptr);
      delete ptr;
    });
  };

  inline auto get() const -> ncclComm_t { return *comm_; }

private:
  std::shared_ptr<ncclComm_t> comm_;
};
}  // namespace spfft

#endif
#endif
