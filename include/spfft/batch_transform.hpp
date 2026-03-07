#ifndef SPFFT_BATCH_TRANSFORM_HPP
#define SPFFT_BATCH_TRANSFORM_HPP

#include <memory>
#include "spfft/config.h"
#include "spfft/types.h"

namespace spfft {

template <typename T>
class SPFFT_NO_EXPORT BatchTransformInternal;

/**
 * A batched transform in double precision. All batches share the same grid dimensions and frequency
 * indices. Local (non-MPI), host-only.
 */
class SPFFT_EXPORT BatchTransform {
public:
  using ValueType = double;

  /**
   * Create a batched transform.
   *
   * @param[in] maxNumThreads The maximum number of threads to use.
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
  BatchTransform(int maxNumThreads, SpfftTransformType transformType, int dimX, int dimY, int dimZ,
                 int batchSize, int numLocalElements, SpfftIndexFormatType indexFormat,
                 const int* indices);

  /**
   * Default copy constructor.
   */
  BatchTransform(const BatchTransform&) = default;

  /**
   * Default move constructor.
   */
  BatchTransform(BatchTransform&&) = default;

  /**
   * Default copy operator.
   */
  BatchTransform& operator=(const BatchTransform&) = default;

  /**
   * Default move operator.
   */
  BatchTransform& operator=(BatchTransform&&) = default;

  /**
   * Execute a forward transform from space domain to frequency domain using internal buffer.
   *
   * @param[out] output Pointer to frequency domain output for all batches.
   *             Layout: batch 0 (2*numLocalElements doubles), then batch 1, etc.
   * @param[in] scaling Controls scaling of output.
   */
  void forward(double* output, SpfftScalingType scaling = SPFFT_NO_SCALING);

  /**
   * Execute a forward transform from space domain to frequency domain.
   *
   * @param[in] input Pointer to space domain input for all batches.
   *            Layout: batch 0 (dimX*dimY*dimZ doubles), then batch 1, etc.
   * @param[out] output Pointer to frequency domain output for all batches.
   * @param[in] scaling Controls scaling of output.
   */
  void forward(const double* input, double* output, SpfftScalingType scaling = SPFFT_NO_SCALING);

  /**
   * Execute a backward transform from frequency domain to space domain, writing to internal buffer.
   *
   * @param[in] input Pointer to frequency domain input for all batches.
   *            Layout: batch 0 (2*numLocalElements doubles), then batch 1, etc.
   */
  void backward(const double* input);

  /**
   * Execute a backward transform from frequency domain to space domain.
   *
   * @param[in] input Pointer to frequency domain input for all batches.
   * @param[out] output Pointer to space domain output for all batches.
   */
  void backward(const double* input, double* output);

  /**
   * Provides access to the space domain data.
   *
   * @return Pointer to space domain data for all batches.
   *         Layout: batch 0 (dimX*dimY*dimZ doubles), then batch 1, etc.
   */
  double* space_domain_data();

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

private:
  std::shared_ptr<BatchTransformInternal<double>> transform_;
};

}  // namespace spfft
#endif
