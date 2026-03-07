#include "spfft/batch_transform_float.hpp"
#include "spfft/batch_transform_float.h"
#include "spfft/batch_transform_internal.hpp"
#include "spfft/exceptions.hpp"

#ifdef SPFFT_SINGLE_PRECISION

namespace spfft {

BatchTransformFloat::BatchTransformFloat(int maxNumThreads,
                                         SpfftProcessingUnitType processingUnit,
                                         SpfftTransformType transformType, int dimX, int dimY,
                                         int dimZ, int batchSize, int numLocalElements,
                                         SpfftIndexFormatType indexFormat, const int* indices) {
  transform_.reset(new BatchTransformInternal<float>(maxNumThreads, processingUnit, transformType,
                                                     dimX, dimY, dimZ, batchSize,
                                                     numLocalElements, indexFormat, indices));
}

void BatchTransformFloat::forward(SpfftProcessingUnitType inputLocation, float* output,
                                  SpfftScalingType scaling) {
  transform_->forward(inputLocation, output, scaling);
}

void BatchTransformFloat::forward(const float* input, float* output, SpfftScalingType scaling) {
  transform_->forward(input, output, scaling);
}

void BatchTransformFloat::backward(const float* input, SpfftProcessingUnitType outputLocation) {
  transform_->backward(input, outputLocation);
}

void BatchTransformFloat::backward(const float* input, float* output) {
  transform_->backward(input, output);
}

float* BatchTransformFloat::space_domain_data(SpfftProcessingUnitType processingUnit) {
  return transform_->space_domain_data(processingUnit);
}

int BatchTransformFloat::batch_size() const { return transform_->batch_size(); }

int BatchTransformFloat::dim_x() const { return transform_->dim_x(); }

int BatchTransformFloat::dim_y() const { return transform_->dim_y(); }

int BatchTransformFloat::dim_z() const { return transform_->dim_z(); }

int BatchTransformFloat::num_local_elements() const { return transform_->num_local_elements(); }

SpfftTransformType BatchTransformFloat::type() const { return transform_->type(); }

int BatchTransformFloat::num_threads() const { return transform_->num_threads(); }

SpfftProcessingUnitType BatchTransformFloat::processing_unit() const {
  return transform_->processing_unit();
}

}  // namespace spfft

extern "C" {

SpfftError spfft_float_batch_transform_create(SpfftFloatBatchTransform* transform,
                                               int maxNumThreads,
                                               SpfftProcessingUnitType processingUnit,
                                               SpfftTransformType transformType, int dimX, int dimY,
                                               int dimZ, int batchSize, int numLocalElements,
                                               SpfftIndexFormatType indexFormat,
                                               const int* indices) {
  try {
    *transform = new spfft::BatchTransformFloat(maxNumThreads, processingUnit, transformType, dimX,
                                                dimY, dimZ, batchSize, numLocalElements,
                                                indexFormat, indices);
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}

SpfftError spfft_float_batch_transform_destroy(SpfftFloatBatchTransform transform) {
  if (!transform) {
    return SpfftError::SPFFT_INVALID_HANDLE_ERROR;
  }
  try {
    delete reinterpret_cast<spfft::BatchTransformFloat*>(transform);
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}

SpfftError spfft_float_batch_transform_forward(SpfftFloatBatchTransform transform,
                                                SpfftProcessingUnitType inputLocation,
                                                float* output, SpfftScalingType scaling) {
  if (!transform) {
    return SpfftError::SPFFT_INVALID_HANDLE_ERROR;
  }
  try {
    reinterpret_cast<spfft::BatchTransformFloat*>(transform)
        ->forward(inputLocation, output, scaling);
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}

SpfftError spfft_float_batch_transform_forward_ptr(SpfftFloatBatchTransform transform,
                                                    const float* input, float* output,
                                                    SpfftScalingType scaling) {
  if (!transform) {
    return SpfftError::SPFFT_INVALID_HANDLE_ERROR;
  }
  try {
    reinterpret_cast<spfft::BatchTransformFloat*>(transform)->forward(input, output, scaling);
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}

SpfftError spfft_float_batch_transform_backward(SpfftFloatBatchTransform transform,
                                                 const float* input,
                                                 SpfftProcessingUnitType outputLocation) {
  if (!transform) {
    return SpfftError::SPFFT_INVALID_HANDLE_ERROR;
  }
  try {
    reinterpret_cast<spfft::BatchTransformFloat*>(transform)->backward(input, outputLocation);
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}

SpfftError spfft_float_batch_transform_backward_ptr(SpfftFloatBatchTransform transform,
                                                     const float* input, float* output) {
  if (!transform) {
    return SpfftError::SPFFT_INVALID_HANDLE_ERROR;
  }
  try {
    reinterpret_cast<spfft::BatchTransformFloat*>(transform)->backward(input, output);
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}

SpfftError spfft_float_batch_transform_get_space_domain(SpfftFloatBatchTransform transform,
                                                         SpfftProcessingUnitType processingUnit,
                                                         float** data) {
  if (!transform) {
    return SpfftError::SPFFT_INVALID_HANDLE_ERROR;
  }
  try {
    *data = reinterpret_cast<spfft::BatchTransformFloat*>(transform)
                ->space_domain_data(processingUnit);
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}

SpfftError spfft_float_batch_transform_batch_size(SpfftFloatBatchTransform transform,
                                                   int* batchSize) {
  if (!transform) {
    return SpfftError::SPFFT_INVALID_HANDLE_ERROR;
  }
  try {
    *batchSize = reinterpret_cast<spfft::BatchTransformFloat*>(transform)->batch_size();
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}

SpfftError spfft_float_batch_transform_dim_x(SpfftFloatBatchTransform transform, int* dimX) {
  if (!transform) {
    return SpfftError::SPFFT_INVALID_HANDLE_ERROR;
  }
  try {
    *dimX = reinterpret_cast<spfft::BatchTransformFloat*>(transform)->dim_x();
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}

SpfftError spfft_float_batch_transform_dim_y(SpfftFloatBatchTransform transform, int* dimY) {
  if (!transform) {
    return SpfftError::SPFFT_INVALID_HANDLE_ERROR;
  }
  try {
    *dimY = reinterpret_cast<spfft::BatchTransformFloat*>(transform)->dim_y();
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}

SpfftError spfft_float_batch_transform_dim_z(SpfftFloatBatchTransform transform, int* dimZ) {
  if (!transform) {
    return SpfftError::SPFFT_INVALID_HANDLE_ERROR;
  }
  try {
    *dimZ = reinterpret_cast<spfft::BatchTransformFloat*>(transform)->dim_z();
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}

}  // extern "C"

#endif
