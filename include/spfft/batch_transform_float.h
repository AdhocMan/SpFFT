#ifndef SPFFT_BATCH_TRANSFORM_FLOAT_H
#define SPFFT_BATCH_TRANSFORM_FLOAT_H

#include "spfft/config.h"
#include "spfft/errors.h"
#include "spfft/types.h"

#ifdef SPFFT_SINGLE_PRECISION

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Float batch transform handle.
 */
typedef void* SpfftFloatBatchTransform;

/**
 * Create a single precision batched transform. All batches share the same grid dimensions and
 * frequency indices. Local (non-MPI).
 *
 * @param[out] transform Handle to the batch transform.
 * @param[in] maxNumThreads The maximum number of threads to use.
 * @param[in] processingUnit The processing unit (SPFFT_PU_HOST or SPFFT_PU_GPU).
 * @param[in] transformType The transform type. Can be SPFFT_TRANS_C2C or SPFFT_TRANS_R2C.
 * @param[in] dimX The dimension in x.
 * @param[in] dimY The dimension in y.
 * @param[in] dimZ The dimension in z.
 * @param[in] batchSize The number of independent transforms to batch.
 * @param[in] numLocalElements The number of elements in frequency domain (per batch).
 * @param[in] indexFormat The index format. Only SPFFT_INDEX_TRIPLETS currently supported.
 * @param[in] indices Pointer to frequency indices. Centered indexing is allowed.
 * @return Error code or SPFFT_SUCCESS.
 */
SPFFT_EXPORT SpfftError spfft_float_batch_transform_create(SpfftFloatBatchTransform* transform,
                                                            int maxNumThreads,
                                                            SpfftProcessingUnitType processingUnit,
                                                            SpfftTransformType transformType,
                                                            int dimX, int dimY, int dimZ,
                                                            int batchSize, int numLocalElements,
                                                            SpfftIndexFormatType indexFormat,
                                                            const int* indices);

/**
 * Destroy a single precision batch transform.
 *
 * @param[in] transform Handle to the batch transform.
 * @return Error code or SPFFT_SUCCESS.
 */
SPFFT_EXPORT SpfftError spfft_float_batch_transform_destroy(SpfftFloatBatchTransform transform);

/**
 * Execute a forward transform from space domain to frequency domain using internal buffer.
 *
 * @param[in] transform Handle to the batch transform.
 * @param[in] inputLocation The processing unit, to take the input from. Can be SPFFT_PU_HOST or
 * SPFFT_PU_GPU (if GPU is set as execution unit).
 * @param[out] output Pointer to frequency domain output for all batches.
 * @param[in] scaling Controls scaling of output.
 * @return Error code or SPFFT_SUCCESS.
 */
SPFFT_EXPORT SpfftError spfft_float_batch_transform_forward(SpfftFloatBatchTransform transform,
                                                             SpfftProcessingUnitType inputLocation,
                                                             float* output,
                                                             SpfftScalingType scaling);

/**
 * Execute a forward transform from space domain to frequency domain.
 *
 * @param[in] transform Handle to the batch transform.
 * @param[in] input Pointer to space domain input for all batches.
 * @param[out] output Pointer to frequency domain output for all batches.
 * @param[in] scaling Controls scaling of output.
 * @return Error code or SPFFT_SUCCESS.
 */
SPFFT_EXPORT SpfftError spfft_float_batch_transform_forward_ptr(SpfftFloatBatchTransform transform,
                                                                 const float* input, float* output,
                                                                 SpfftScalingType scaling);

/**
 * Execute a backward transform from frequency domain to space domain, writing to internal buffer.
 *
 * @param[in] transform Handle to the batch transform.
 * @param[in] input Pointer to frequency domain input for all batches.
 * @param[in] outputLocation The processing unit, to place the output at. Can be SPFFT_PU_HOST or
 * SPFFT_PU_GPU (if GPU is set as execution unit).
 * @return Error code or SPFFT_SUCCESS.
 */
SPFFT_EXPORT SpfftError spfft_float_batch_transform_backward(SpfftFloatBatchTransform transform,
                                                              const float* input,
                                                              SpfftProcessingUnitType outputLocation);

/**
 * Execute a backward transform from frequency domain to space domain.
 *
 * @param[in] transform Handle to the batch transform.
 * @param[in] input Pointer to frequency domain input for all batches.
 * @param[out] output Pointer to space domain output for all batches.
 * @return Error code or SPFFT_SUCCESS.
 */
SPFFT_EXPORT SpfftError spfft_float_batch_transform_backward_ptr(SpfftFloatBatchTransform transform,
                                                                  const float* input,
                                                                  float* output);

/**
 * Get the space domain data pointer.
 *
 * @param[in] transform Handle to the batch transform.
 * @param[in] processingUnit The processing unit (SPFFT_PU_HOST or SPFFT_PU_GPU).
 * @param[out] data Pointer to space domain data for all batches.
 * @return Error code or SPFFT_SUCCESS.
 */
SPFFT_EXPORT SpfftError spfft_float_batch_transform_get_space_domain(
    SpfftFloatBatchTransform transform, SpfftProcessingUnitType processingUnit, float** data);

/**
 * @param[in] transform Handle to the batch transform.
 * @param[out] batchSize Number of batches.
 * @return Error code or SPFFT_SUCCESS.
 */
SPFFT_EXPORT SpfftError spfft_float_batch_transform_batch_size(SpfftFloatBatchTransform transform,
                                                                int* batchSize);

/**
 * @param[in] transform Handle to the batch transform.
 * @param[out] dimX Dimension in x.
 * @return Error code or SPFFT_SUCCESS.
 */
SPFFT_EXPORT SpfftError spfft_float_batch_transform_dim_x(SpfftFloatBatchTransform transform,
                                                           int* dimX);

/**
 * @param[in] transform Handle to the batch transform.
 * @param[out] dimY Dimension in y.
 * @return Error code or SPFFT_SUCCESS.
 */
SPFFT_EXPORT SpfftError spfft_float_batch_transform_dim_y(SpfftFloatBatchTransform transform,
                                                           int* dimY);

/**
 * @param[in] transform Handle to the batch transform.
 * @param[out] dimZ Dimension in z.
 * @return Error code or SPFFT_SUCCESS.
 */
SPFFT_EXPORT SpfftError spfft_float_batch_transform_dim_z(SpfftFloatBatchTransform transform,
                                                           int* dimZ);

#ifdef __cplusplus
}
#endif

#endif

#endif
