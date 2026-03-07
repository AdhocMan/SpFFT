#include "spfft/batch_transform.hpp"
#include "spfft/batch_transform.h"
#include "spfft/batch_transform_internal.hpp"
#include "spfft/exceptions.hpp"

namespace spfft {

BatchTransform::BatchTransform(int maxNumThreads, SpfftTransformType transformType, int dimX,
                               int dimY, int dimZ, int batchSize, int numLocalElements,
                               SpfftIndexFormatType indexFormat, const int* indices) {
  transform_.reset(new BatchTransformInternal<double>(maxNumThreads, transformType, dimX, dimY, dimZ,
                                                      batchSize, numLocalElements, indexFormat,
                                                      indices));
}

void BatchTransform::forward(double* output, SpfftScalingType scaling) {
  transform_->forward(output, scaling);
}

void BatchTransform::forward(const double* input, double* output, SpfftScalingType scaling) {
  transform_->forward(input, output, scaling);
}

void BatchTransform::backward(const double* input) { transform_->backward(input); }

void BatchTransform::backward(const double* input, double* output) {
  transform_->backward(input, output);
}

double* BatchTransform::space_domain_data() { return transform_->space_domain_data(); }

int BatchTransform::batch_size() const { return transform_->batch_size(); }

int BatchTransform::dim_x() const { return transform_->dim_x(); }

int BatchTransform::dim_y() const { return transform_->dim_y(); }

int BatchTransform::dim_z() const { return transform_->dim_z(); }

int BatchTransform::num_local_elements() const { return transform_->num_local_elements(); }

SpfftTransformType BatchTransform::type() const { return transform_->type(); }

int BatchTransform::num_threads() const { return transform_->num_threads(); }

}  // namespace spfft

extern "C" {

SpfftError spfft_batch_transform_create(SpfftBatchTransform* transform, int maxNumThreads,
                                         SpfftTransformType transformType, int dimX, int dimY,
                                         int dimZ, int batchSize, int numLocalElements,
                                         SpfftIndexFormatType indexFormat, const int* indices) {
  try {
    *transform = new spfft::BatchTransform(maxNumThreads, transformType, dimX, dimY, dimZ,
                                            batchSize, numLocalElements, indexFormat, indices);
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}

SpfftError spfft_batch_transform_destroy(SpfftBatchTransform transform) {
  if (!transform) {
    return SpfftError::SPFFT_INVALID_HANDLE_ERROR;
  }
  try {
    delete reinterpret_cast<spfft::BatchTransform*>(transform);
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}

SpfftError spfft_batch_transform_forward(SpfftBatchTransform transform, double* output,
                                          SpfftScalingType scaling) {
  if (!transform) {
    return SpfftError::SPFFT_INVALID_HANDLE_ERROR;
  }
  try {
    reinterpret_cast<spfft::BatchTransform*>(transform)->forward(output, scaling);
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}

SpfftError spfft_batch_transform_forward_ptr(SpfftBatchTransform transform, const double* input,
                                              double* output, SpfftScalingType scaling) {
  if (!transform) {
    return SpfftError::SPFFT_INVALID_HANDLE_ERROR;
  }
  try {
    reinterpret_cast<spfft::BatchTransform*>(transform)->forward(input, output, scaling);
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}

SpfftError spfft_batch_transform_backward(SpfftBatchTransform transform, const double* input) {
  if (!transform) {
    return SpfftError::SPFFT_INVALID_HANDLE_ERROR;
  }
  try {
    reinterpret_cast<spfft::BatchTransform*>(transform)->backward(input);
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}

SpfftError spfft_batch_transform_backward_ptr(SpfftBatchTransform transform, const double* input,
                                               double* output) {
  if (!transform) {
    return SpfftError::SPFFT_INVALID_HANDLE_ERROR;
  }
  try {
    reinterpret_cast<spfft::BatchTransform*>(transform)->backward(input, output);
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}

SpfftError spfft_batch_transform_get_space_domain(SpfftBatchTransform transform, double** data) {
  if (!transform) {
    return SpfftError::SPFFT_INVALID_HANDLE_ERROR;
  }
  try {
    *data = reinterpret_cast<spfft::BatchTransform*>(transform)->space_domain_data();
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}

SpfftError spfft_batch_transform_batch_size(SpfftBatchTransform transform, int* batchSize) {
  if (!transform) {
    return SpfftError::SPFFT_INVALID_HANDLE_ERROR;
  }
  try {
    *batchSize = reinterpret_cast<spfft::BatchTransform*>(transform)->batch_size();
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}

SpfftError spfft_batch_transform_dim_x(SpfftBatchTransform transform, int* dimX) {
  if (!transform) {
    return SpfftError::SPFFT_INVALID_HANDLE_ERROR;
  }
  try {
    *dimX = reinterpret_cast<spfft::BatchTransform*>(transform)->dim_x();
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}

SpfftError spfft_batch_transform_dim_y(SpfftBatchTransform transform, int* dimY) {
  if (!transform) {
    return SpfftError::SPFFT_INVALID_HANDLE_ERROR;
  }
  try {
    *dimY = reinterpret_cast<spfft::BatchTransform*>(transform)->dim_y();
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}

SpfftError spfft_batch_transform_dim_z(SpfftBatchTransform transform, int* dimZ) {
  if (!transform) {
    return SpfftError::SPFFT_INVALID_HANDLE_ERROR;
  }
  try {
    *dimZ = reinterpret_cast<spfft::BatchTransform*>(transform)->dim_z();
  } catch (const spfft::GenericError& e) {
    return e.error_code();
  } catch (...) {
    return SpfftError::SPFFT_UNKNOWN_ERROR;
  }
  return SpfftError::SPFFT_SUCCESS;
}

}  // extern "C"
