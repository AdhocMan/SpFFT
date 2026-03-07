#include <algorithm>
#include <cmath>
#include <complex>
#include <random>
#include <vector>
#include "gtest/gtest.h"
#include "spfft/batch_transform.hpp"
#include "spfft/config.h"
#include "spfft/transform.hpp"
#include "test_util/generate_indices.hpp"
#include "util/common_types.hpp"

using namespace spfft;

class TestBatchTransform
    : public ::testing::TestWithParam<
          std::tuple<int, int, int, int, SpfftTransformType, SpfftProcessingUnitType>> {
protected:
  int dimX_ = std::get<0>(GetParam());
  int dimY_ = std::get<1>(GetParam());
  int dimZ_ = std::get<2>(GetParam());
  int batchSize_ = std::get<3>(GetParam());
  SpfftTransformType transformType_ = std::get<4>(GetParam());
  SpfftProcessingUnitType processingUnit_ = std::get<5>(GetParam());
};

TEST_P(TestBatchTransform, BackwardBatchVsSingle) {
  std::mt19937 randGen(42);
  std::uniform_real_distribution<double> uniformRandDis(0.0, 1.0);

  const bool hermitian = (transformType_ == SPFFT_TRANS_R2C);
  std::vector<double> zStickDistribution(1, 1.0);
  auto valueIndicesPerRank =
      create_value_indices(randGen, zStickDistribution, 0.7, 0.7, dimX_, dimY_, dimZ_, hermitian);
  const auto& indices = valueIndicesPerRank[0];
  const int numLocalElements = indices.size() / 3;

  if (numLocalElements == 0) return;

  BatchTransform batchTransform(1, processingUnit_, transformType_, dimX_, dimY_, dimZ_, batchSize_,
                                numLocalElements, SPFFT_INDEX_TRIPLETS, indices.data());

  // For C2C, space domain contains complex values (2 doubles each)
  const int spaceDomainDoubles =
      (transformType_ == SPFFT_TRANS_C2C) ? 2 * dimX_ * dimY_ * dimZ_ : dimX_ * dimY_ * dimZ_;

  // Generate frequency domain input for each batch
  std::vector<double> batchedFreqInput(batchSize_ * 2 * numLocalElements);
  std::vector<std::vector<double>> freqInputs(batchSize_);
  for (int b = 0; b < batchSize_; ++b) {
    freqInputs[b].resize(2 * numLocalElements);
    for (int i = 0; i < 2 * numLocalElements; ++i) {
      double val = uniformRandDis(randGen);
      freqInputs[b][i] = val;
      batchedFreqInput[b * 2 * numLocalElements + i] = val;
    }
  }

  // Backward with batch transform
  batchTransform.backward(batchedFreqInput.data(), SPFFT_PU_HOST);
  double* batchedSpace = batchTransform.space_domain_data(SPFFT_PU_HOST);

  // Compare with individual transforms
  for (int b = 0; b < batchSize_; ++b) {
    Transform singleTransform(1, SPFFT_PU_HOST, transformType_, dimX_, dimY_, dimZ_,
                              numLocalElements, SPFFT_INDEX_TRIPLETS, indices.data());
    singleTransform.backward(freqInputs[b].data(), SPFFT_PU_HOST);
    double* singleSpace = singleTransform.space_domain_data(SPFFT_PU_HOST);

    for (int i = 0; i < spaceDomainDoubles; ++i) {
      ASSERT_NEAR(batchedSpace[b * spaceDomainDoubles + i], singleSpace[i], 1e-6)
          << "Batch " << b << ", element " << i;
    }
  }
}

TEST_P(TestBatchTransform, ForwardBatchVsSingle) {
  std::mt19937 randGen(42);
  std::uniform_real_distribution<double> uniformRandDis(0.0, 1.0);

  const bool hermitian = (transformType_ == SPFFT_TRANS_R2C);
  std::vector<double> zStickDistribution(1, 1.0);
  auto valueIndicesPerRank =
      create_value_indices(randGen, zStickDistribution, 0.7, 0.7, dimX_, dimY_, dimZ_, hermitian);
  const auto& indices = valueIndicesPerRank[0];
  const int numLocalElements = indices.size() / 3;

  if (numLocalElements == 0) return;

  BatchTransform batchTransform(1, processingUnit_, transformType_, dimX_, dimY_, dimZ_, batchSize_,
                                numLocalElements, SPFFT_INDEX_TRIPLETS, indices.data());

  // For C2C, space domain contains complex values (2 doubles each)
  const int spaceDomainDoubles =
      (transformType_ == SPFFT_TRANS_C2C) ? 2 * dimX_ * dimY_ * dimZ_ : dimX_ * dimY_ * dimZ_;

  // Fill each batch with distinct space domain data
  double* batchedSpace = batchTransform.space_domain_data(SPFFT_PU_HOST);
  std::vector<std::vector<double>> spaceInputs(batchSize_);
  for (int b = 0; b < batchSize_; ++b) {
    spaceInputs[b].resize(spaceDomainDoubles);
    for (int i = 0; i < spaceDomainDoubles; ++i) {
      double val = uniformRandDis(randGen);
      batchedSpace[b * spaceDomainDoubles + i] = val;
      spaceInputs[b][i] = val;
    }
  }

  // Forward with batch transform
  std::vector<double> batchedFreqOutput(batchSize_ * 2 * numLocalElements);
  batchTransform.forward(SPFFT_PU_HOST, batchedFreqOutput.data(), SPFFT_NO_SCALING);

  // Compare with individual transforms
  for (int b = 0; b < batchSize_; ++b) {
    Transform singleTransform(1, SPFFT_PU_HOST, transformType_, dimX_, dimY_, dimZ_,
                              numLocalElements, SPFFT_INDEX_TRIPLETS, indices.data());

    // Copy space domain data to single transform
    double* singleSpace = singleTransform.space_domain_data(SPFFT_PU_HOST);
    std::copy(spaceInputs[b].begin(), spaceInputs[b].end(), singleSpace);

    std::vector<double> singleFreqOutput(2 * numLocalElements);
    singleTransform.forward(SPFFT_PU_HOST, singleFreqOutput.data(), SPFFT_NO_SCALING);

    for (int i = 0; i < 2 * numLocalElements; ++i) {
      ASSERT_NEAR(batchedFreqOutput[b * 2 * numLocalElements + i], singleFreqOutput[i], 1e-6)
          << "Batch " << b << ", element " << i;
    }
  }
}

TEST_P(TestBatchTransform, BackwardForwardRoundTrip) {
  // Round trip only works in backward->forward direction (freq domain round trip)
  // because compression/decompression only stores a subset of frequencies.
  // For R2C, Hermitian symmetry constraints make freq-domain round trip unreliable
  // with arbitrary input, so we skip R2C (covered by BackwardBatchVsSingle + ForwardBatchVsSingle).
  if (transformType_ == SPFFT_TRANS_R2C) return;

  std::mt19937 randGen(42);
  std::uniform_real_distribution<double> uniformRandDis(0.0, 1.0);

  const bool hermitian = (transformType_ == SPFFT_TRANS_R2C);
  std::vector<double> zStickDistribution(1, 1.0);
  auto valueIndicesPerRank =
      create_value_indices(randGen, zStickDistribution, 0.7, 0.7, dimX_, dimY_, dimZ_, hermitian);
  const auto& indices = valueIndicesPerRank[0];
  const int numLocalElements = indices.size() / 3;

  if (numLocalElements == 0) return;

  BatchTransform batchTransform(1, processingUnit_, transformType_, dimX_, dimY_, dimZ_, batchSize_,
                                numLocalElements, SPFFT_INDEX_TRIPLETS, indices.data());

  // Generate frequency domain input
  std::vector<double> freqInput(batchSize_ * 2 * numLocalElements);
  for (auto& v : freqInput) v = uniformRandDis(randGen);
  std::vector<double> freqInputCopy = freqInput;

  // Backward then forward with full scaling should recover the input
  batchTransform.backward(freqInput.data(), SPFFT_PU_HOST);

  std::vector<double> freqOutput(batchSize_ * 2 * numLocalElements);
  batchTransform.forward(SPFFT_PU_HOST, freqOutput.data(), SPFFT_FULL_SCALING);

  for (int i = 0; i < static_cast<int>(freqOutput.size()); ++i) {
    ASSERT_NEAR(freqOutput[i], freqInputCopy[i], 1e-6) << "Element " << i;
  }
}

static auto batch_param_names(
    const ::testing::TestParamInfo<
        std::tuple<int, int, int, int, SpfftTransformType, SpfftProcessingUnitType>>& info)
    -> std::string {
  std::string name;
  name += (std::get<5>(info.param) == SPFFT_PU_HOST) ? "Host" : "GPU";
  name += "_";
  name += (std::get<4>(info.param) == SPFFT_TRANS_C2C) ? "C2C" : "R2C";
  name += "_";
  name += std::to_string(std::get<0>(info.param));
  name += "x";
  name += std::to_string(std::get<1>(info.param));
  name += "x";
  name += std::to_string(std::get<2>(info.param));
  name += "_B";
  name += std::to_string(std::get<3>(info.param));
  return name;
}

INSTANTIATE_TEST_SUITE_P(
    HostTest, TestBatchTransform,
    ::testing::Combine(::testing::Values(1, 2, 11, 12, 13),
                       ::testing::Values(1, 2, 11, 12, 13),
                       ::testing::Values(1, 2, 11, 12, 13),
                       ::testing::Values(1, 2, 4),
                       ::testing::Values(SPFFT_TRANS_C2C, SPFFT_TRANS_R2C),
                       ::testing::Values(SPFFT_PU_HOST)),
    batch_param_names);

#if defined(SPFFT_CUDA) || defined(SPFFT_ROCM)
INSTANTIATE_TEST_SUITE_P(
    GPUTest, TestBatchTransform,
    ::testing::Combine(::testing::Values(1, 2, 11, 12, 13),
                       ::testing::Values(1, 2, 11, 12, 13),
                       ::testing::Values(1, 2, 11, 12, 13),
                       ::testing::Values(1, 2, 4),
                       ::testing::Values(SPFFT_TRANS_C2C, SPFFT_TRANS_R2C),
                       ::testing::Values(SPFFT_PU_GPU)),
    batch_param_names);
#endif
