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
#ifndef SPFFT_LOCAL_TRANSPOSE_BATCH_KERNELS_HPP
#define SPFFT_LOCAL_TRANSPOSE_BATCH_KERNELS_HPP

#include "gpu_util/gpu_fft_api.hpp"
#include "memory/gpu_array_view.hpp"

namespace spfft {

// ------------------
// Backward
// ------------------

auto local_transpose_batch_backward(
    const gpu::StreamType stream, const GPUArrayView1D<int>& indices,
    const typename gpu::fft::ComplexType<double>::type* freqZData,
    typename gpu::fft::ComplexType<double>::type* spaceDomainFlat, const int numZSticks,
    const int dimZ, const int xyPlaneSize, const int batchSize) -> void;

auto local_transpose_batch_backward(
    const gpu::StreamType stream, const GPUArrayView1D<int>& indices,
    const typename gpu::fft::ComplexType<float>::type* freqZData,
    typename gpu::fft::ComplexType<float>::type* spaceDomainFlat, const int numZSticks,
    const int dimZ, const int xyPlaneSize, const int batchSize) -> void;

// ------------------
// Forward
// ------------------

auto local_transpose_batch_forward(
    const gpu::StreamType stream, const GPUArrayView1D<int>& indices,
    const typename gpu::fft::ComplexType<double>::type* spaceDomainFlat,
    typename gpu::fft::ComplexType<double>::type* freqZData, const int numZSticks, const int dimZ,
    const int xyPlaneSize, const int batchSize) -> void;

auto local_transpose_batch_forward(
    const gpu::StreamType stream, const GPUArrayView1D<int>& indices,
    const typename gpu::fft::ComplexType<float>::type* spaceDomainFlat,
    typename gpu::fft::ComplexType<float>::type* freqZData, const int numZSticks, const int dimZ,
    const int xyPlaneSize, const int batchSize) -> void;

}  // namespace spfft
#endif
