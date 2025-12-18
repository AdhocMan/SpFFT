#include <algorithm>
#include <chrono>
#include <complex>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <thread>
#include <vector>

#include "compression/compression_gpu.hpp"
#include "compression/compression_host.hpp"
#include "fft/transform_1d_gpu.hpp"
#include "fft/transform_1d_host.hpp"
#include "fft/transform_2d_gpu.hpp"
#include "fft/transform_real_2d_gpu.hpp"
#include "memory/array_view_utility.hpp"
#include "memory/host_array.hpp"
#include "parameters/parameters.hpp"
#include "spfft/config.h"
#include "spfft/spfft.hpp"
#include "timing/timing.hpp"
#include "transpose/transpose.hpp"
#include "transpose/transpose_gpu.hpp"
#include "util/omp_definitions.hpp"

#if defined(SPFFT_CUDA) || defined(SPFFT_ROCM)
#include "gpu_util/gpu_runtime_api.hpp"
#include "gpu_util/gpu_transfer.hpp"
#include "memory/gpu_array.hpp"
#endif

#ifdef SPFFT_MPI
#include <mpi.h>
#include "mpi_util/mpi_communicator_handle.hpp"
#include "mpi_util/mpi_init_handle.hpp"
#endif

// external dependencies
#include "CLI/CLI.hpp"
#include "nlohmann/json.hpp"

// #include <unistd.h>  // for MPI debugging

using namespace spfft;

std::vector<int> generate_spherical_indices(int dimX, int dimY, int dimZ,int radius) {
  if (radius > dimX / 2 || radius > dimY / 2 || radius > dimZ / 2) {
    throw std::runtime_error("radiues too large");
  }


  std::vector<int> indices;
  indices.reserve(1000);
  for (int x = -radius; x <  radius; ++x) {
    for (int y = -radius; y < radius; ++y) {
      for (int z = -radius; z < radius; ++z) {
        if (x * x + y * y + z * z <= radius * radius) {
          indices.emplace_back(x + dimX / 2);
          indices.emplace_back(y + dimY / 2);
          indices.emplace_back(z + dimZ / 2);
        }
      }
    }
  }

  return indices;
}


class BatchTransform {
  public:
    BatchTransform(int batchSize, int dimX, int dimY, int dimZ, int nIndices, const int* indices)
        : batchSize_(batchSize) {
      fftWorkBuffer_.reset(new GPUArray<char>());
      param_ = std::make_shared<Parameters>(SPFFT_TRANS_C2C, dimX, dimY, dimZ, nIndices,
                                            SPFFT_INDEX_TRIPLETS, indices);

      gpuArray1_ = GPUArray<typename gpu::fft::ComplexType<double>::type>(
          static_cast<SizeType>(param_->dim_x() * param_->dim_y() * param_->dim_z()));
      gpuArray2_ = GPUArray<typename gpu::fft::ComplexType<double>::type>(
          static_cast<SizeType>(param_->dim_x() * param_->dim_y() * param_->dim_z()));

      const SizeType numLocalZSticks = param_->num_z_sticks(0);

      freqDomainDataGPU_ =
          create_3d_view(gpuArray1_, 0, batchSize, numLocalZSticks, param_->dim_z());
      freqDomainFFTDataGPU_ =
          create_2d_view(gpuArray1_, 0, batchSize * numLocalZSticks, param_->dim_z());

      compression_.reset(new CompressionGPU(param_));

      transformZ_ = std::unique_ptr<TransformGPU>(
          new Transform1DGPU<double>(freqDomainFFTDataGPU_, stream_, fftWorkBuffer_));

      freqDomainXYGPU_ = create_3d_view(gpuArray2_, 0, batchSize * param_->dim_z(), param_->dim_y(),
                                        param_->dim_x_freq());  // must not overlap with z-sticks
      // transpose_.reset(
      //     new TransposeGPU<double>(param_, stream_, freqDomainXYGPU_, freqDomainDataGPU_));

      const auto zStickXYIndices = param_->z_stick_xy_indices(0);

      std::vector<int> transposedIndices;
      transposedIndices.reserve(zStickXYIndices.size());

      for (const auto& index : zStickXYIndices) {
        const int x = index / param_->dim_y();
        const int y = index - x * param_->dim_y();
        transposedIndices.emplace_back(y * param_->dim_x_freq() + x);
      }

      transposeIndicesGPU_ = GPUArray<int>(transposedIndices.size());
      copy_to_gpu(transposedIndices, transposeIndicesGPU_);

      transformXY_ = std::unique_ptr<TransformGPU>(
          new Transform2DGPU<double>(freqDomainXYGPU_, stream_, fftWorkBuffer_));
    }

    void backward(const double* input, double* output) {
      compression_->decompress_batch(stream_, input, freqDomainDataGPU_);
      transformZ_->backward();

      // transpose
      gpu::check_status(gpu::memset_async(
          static_cast<void*>(freqDomainXYGPU_.data()), 0,
          freqDomainXYGPU_.size() * sizeof(typename decltype(freqDomainXYGPU_)::ValueType),
          stream_.get()));
      local_transpose_batched_backward(
          stream_.get(),
          GPUArrayView1D<int>(transposeIndicesGPU_.data(), transposeIndicesGPU_.size(),
                              transposeIndicesGPU_.device_id()),
          freqDomainDataGPU_, freqDomainXYGPU_, batchSize_);
      // transpose_->backward();
      


      transformXY_->backward(freqDomainXYGPU_.data(), output);
    }

  private:
    int batchSize_;
    GPUStreamHandle stream_ = GPUStreamHandle(0);
    GPUArray<typename gpu::fft::ComplexType<double>::type> gpuArray1_;
    GPUArray<typename gpu::fft::ComplexType<double>::type> gpuArray2_;
    std::shared_ptr<GPUArray<char>> fftWorkBuffer_;

    std::shared_ptr<Parameters> param_;

    std::unique_ptr<TransformGPU> transformZ_;
    // std::unique_ptr<Transpose> transpose_;
    std::unique_ptr<TransformGPU> transformXY_;
    std::unique_ptr<CompressionGPU> compression_;

    GPUArray<int> transposeIndicesGPU_;

    GPUArrayView2D<typename gpu::fft::ComplexType<double>::type> freqDomainFFTDataGPU_;
    GPUArrayView3D<typename gpu::fft::ComplexType<double>::type> freqDomainDataGPU_;
    GPUArrayView3D<typename gpu::fft::ComplexType<double>::type> freqDomainXYGPU_;
};



void transform_batch(int batchSize, int dimX, int dimY, int dimZ,int radius, const double* input, double* output) {

  auto indices = generate_spherical_indices(dimX, dimY, dimZ, radius);

  auto param = std::make_shared<Parameters>(SPFFT_TRANS_C2C, dimX, dimY, dimZ, indices.size() / 3, SPFFT_INDEX_TRIPLETS,
                   indices.data());


  HostArray<std::complex<double>> array1(dimX * dimY * dimZ);
  HostArray<std::complex<double>> array2(dimX * dimY * dimZ);



  const SizeType numLocalZSticks = param->num_z_sticks(0);
  const SizeType numLocalXYPlanes = param->num_xy_planes(0);

  auto freqDomainData = create_2d_view(array1, 0, numLocalZSticks, param->dim_z());

  CompressionHost compression(param);

  compression.decompress(input, freqDomainData);

}



int main(int argc, char** argv) {

  int dimX = 200;
  int dimY = 200;
  int dimZ = 200;

  int radius = 50;

  int batchSize = 1;

  auto indices = generate_spherical_indices(dimX, dimY, dimZ, radius);
  auto nIndices = indices.size() / 3;

  std::vector<double> input(batchSize * 2 * nIndices);
  std::vector<double> output(batchSize * 2 * dimX * dimY * dimZ);
  std::vector<double> outputRef(batchSize * 2 * dimX * dimY * dimZ);


  std::minstd_rand randGen(42);
  std::uniform_real_distribution<double> dist(0.1, 1.0);

  for(auto& val : input) {
    val = dist(randGen);
  }


  GPUStreamHandle stream = GPUStreamHandle(0);

  BatchTransform bt(batchSize, dimX, dimY, dimZ, nIndices, indices.data());
  Transform transform(1, SPFFT_PU_GPU, SPFFT_TRANS_C2C, dimX, dimY, dimZ, nIndices, SPFFT_INDEX_TRIPLETS, indices.data());

  GPUArray<double> gpuInput(input.size());
  GPUArray<double> gpuOutput(output.size());
  GPUArray<double> gpuOutputRef(output.size());

  gpu::check_status(gpu::memcpy_async(gpuInput.data(), input.data(), input.size() * sizeof(double),
                                      gpu::flag::MemcpyHostToDevice, stream.get()));

  gpu::check_status(gpu::stream_synchronize(stream.get()));


  bt.backward(gpuInput.data(), gpuOutput.data());

  transform.backward(gpuInput.data(), gpuOutputRef.data());



  gpu::check_status(gpu::stream_synchronize(stream.get()));
  gpu::check_status(gpu::memcpy_async(output.data(), gpuOutput.data(),
                                      output.size() * sizeof(double), gpu::flag::MemcpyDeviceToHost,
                                      stream.get()));
  gpu::check_status(gpu::memcpy_async(outputRef.data(), gpuOutputRef.data(),
                                      output.size() * sizeof(double), gpu::flag::MemcpyDeviceToHost,
                                      stream.get()));
  gpu::check_status(gpu::stream_synchronize(stream.get()));


  // compare

  for(std::size_t i = 0; i < output.size(); ++i) {
    if(std::abs(output[i] - outputRef[i]) > 0.0001) {
      std::cout << "Error: output = " << output[i] << ", ref = " << outputRef[i] << std::endl;
    }
  }

}
