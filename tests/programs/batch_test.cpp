#include <algorithm>
#include <chrono>
#include <complex>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <thread>
#include <vector>
#include "fft/transform_1d_host.hpp"
#include "memory/array_view_utility.hpp"
#include "memory/host_array.hpp"
#include "spfft/config.h"
#include "spfft/spfft.hpp"
#include "timing/timing.hpp"
#include "compression/compression_host.hpp"
#include "util/omp_definitions.hpp"
#include "parameters/parameters.hpp"

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



void transform_batch(int batchsize, int dimX, int dimY, int dimZ,int radius, const double* input, double* output) {

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

  int batchsize = 1;

  std::vector<double> input(2 * dimX * dimY * dimZ);
  std::vector<double> output(2 * dimX * dimY * dimZ);

  transform_batch(batchsize, dimX, dimY, dimZ, radius, input.data(), output.data());

}
