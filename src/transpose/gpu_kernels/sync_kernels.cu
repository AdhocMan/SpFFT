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
#include "gpu_util/gpu_runtime.hpp"
#include "gpu_util/gpu_runtime_api.hpp"
#include "transpose/gpu_kernels/sync_kernels.hpp"

namespace spfft {

__global__ static void signal_value_kernel(volatile unsigned int* addr, unsigned int value) {
  if (threadIdx.x == 0) {
   // atomicExch_system(addr, value);
   __threadfence_system();
   atomicExch_system(addr, value);
  }
}

auto signal_value(const gpu::StreamType& stream, unsigned int* addr, unsigned int value) -> void {
  const dim3 threadBlock(1);
  const dim3 threadGrid(1);
  launch_kernel(signal_value_kernel, threadGrid, threadBlock, 0, stream, addr, value);
}

__global__ static void wait_for_value_kernel(volatile unsigned int* addr, unsigned int expected) {
  if (threadIdx.x == 0) {
   while (*addr != expected) {
   }
  }
}

auto wait_for_value(const gpu::StreamType& stream, unsigned int* addr, unsigned int expected) -> void {
  const dim3 threadBlock(1);
  const dim3 threadGrid(1);
  launch_kernel(wait_for_value_kernel, threadGrid, threadBlock, 0, stream, addr, expected);
}


}  // namespace spfft
