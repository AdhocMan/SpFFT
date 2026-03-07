#ifndef SPFFT_BATCH_TRANSFORM_FLOAT_HPP
#define SPFFT_BATCH_TRANSFORM_FLOAT_HPP

#include <memory>
#include "spfft/config.h"
#include "spfft/types.h"

namespace spfft {

template <typename T>
class SPFFT_NO_EXPORT BatchTransformInternal;

#ifdef SPFFT_SINGLE_PRECISION

/**
 * A batched transform in single precision. All batches share the same grid dimensions and frequency
 * indices. Local (non-MPI).
 */
class SPFFT_EXPORT BatchTransformFloat {
public:
  using ValueType = float;

  /**
   * Create a batched transform.
   *
   * @param[in] maxNumThreads The maximum number of threads to use.
   * @param[in] processingUnit The processing unit (SPFFT_PU_HOST or SPFFT_PU_GPU).
   * @param[in] transformType The transform type (complex to complex or real to complex). Can be
   * SPFFT_TRANS_C2C or SPFFT_TRANS_R2C.
   * @param[in] dimX The dimension in x.
   * @param[in] dimY The dimension in y.
   * @param[in] dimZ The dimension in z.
   * @param[in] batchSize The number of independent transforms to batch.
   * @param[in] numLocalElements The number of elements in frequency domain (per batch).
   * @param[in] indexFormat The index format. Only SPFFT_INDEX_TRIPLETS currently supported.
   * @param[in] indices Pointer to frequency indices. Centered indexing is allowed.
   */
  BatchTransformFloat(int maxNumThreads, SpfftProcessingUnitType processingUnit,
                      SpfftTransformType transformType, int dimX, int dimY, int dimZ, int batchSize,
                      int numLocalElements, SpfftIndexFormatType indexFormat, const int* indices);

  /**
   * Default copy constructor.
   */
  BatchTransformFloat(const BatchTransformFloat&) = default;

  /**
   * Default move constructor.
   */
  BatchTransformFloat(BatchTransformFloat&&) = default;

  /**
   * Default copy operator.
   */
  BatchTransformFloat& operator=(const BatchTransformFloat&) = default;

  /**
   * Default move operator.
   */
  BatchTransformFloat& operator=(BatchTransformFloat&&) = default;

  /**
   * Execute a forward transform from space domain to frequency domain using internal buffer.
   *
   * @param[in] inputLocation The processing unit to take the input from. Can be SPFFT_PU_HOST or
   * SPFFT_PU_GPU (if GPU is set as execution unit).
   * @param[out] output Pointer to frequency domain output for all batches.
   *             Layout: batch 0 (2*numLocalElements floats), then batch 1, etc.
   * @param[in] scaling Controls scaling of output.
   */
  void forward(SpfftProcessingUnitType inputLocation, float* output,
               SpfftScalingType scaling = SPFFT_NO_SCALING);

  /**
   * Execute a forward transform from space domain to frequency domain.
   *
   * @param[in] input Pointer to space domain input for all batches.
   *            Layout: batch 0 (dimX*dimY*dimZ floats), then batch 1, etc.
   * @param[out] output Pointer to frequency domain output for all batches.
   * @param[in] scaling Controls scaling of output.
   */
  void forward(const float* input, float* output, SpfftScalingType scaling = SPFFT_NO_SCALING);

  /**
   * Execute a backward transform from frequency domain to space domain, writing to internal buffer.
   *
   * @param[in] input Pointer to frequency domain input for all batches.
   *            Layout: batch 0 (2*numLocalElements floats), then batch 1, etc.
   * @param[in] outputLocation The processing unit to place the output at. Can be SPFFT_PU_HOST or
   * SPFFT_PU_GPU (if GPU is set as execution unit).
   */
  void backward(const float* input, SpfftProcessingUnitType outputLocation);

  /**
   * Execute a backward transform from frequency domain to space domain.
   *
   * @param[in] input Pointer to frequency domain input for all batches.
   * @param[out] output Pointer to space domain output for all batches.
   */
  void backward(const float* input, float* output);

  /**
   * Provides access to the space domain data.
   *
   * @param[in] processingUnit The processing unit to get data from (SPFFT_PU_HOST or SPFFT_PU_GPU).
   * @return Pointer to space domain data for all batches.
   *         Layout: batch 0 (dimX*dimY*dimZ floats), then batch 1, etc.
   */
  float* space_domain_data(SpfftProcessingUnitType processingUnit);

  /**
   * @return Number of batches.
   */
  int batch_size() const;

  /**
   * @return Dimension in x.
   */
  int dim_x() const;

  /**
   * @return Dimension in y.
   */
  int dim_y() const;

  /**
   * @return Dimension in z.
   */
  int dim_z() const;

  /**
   * @return Number of local frequency domain elements per batch.
   */
  int num_local_elements() const;

  /**
   * @return Type of transform.
   */
  SpfftTransformType type() const;

  /**
   * @return Number of threads used.
   */
  int num_threads() const;

  /**
   * @return Processing unit type.
   */
  SpfftProcessingUnitType processing_unit() const;

private:
  std::shared_ptr<BatchTransformInternal<float>> transform_;
};
#endif

}  // namespace spfft
#endif
