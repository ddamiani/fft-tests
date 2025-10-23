#include "CudaErrors.hh"
#include <iostream>

static const char* cufftGetErrorString(cufftResult err) {
  switch (err) {
  case CUFFT_SUCCESS:
    return "CUFFT_SUCCESS";
  case CUFFT_INVALID_PLAN:
      return "CUFFT_INVALID_PLAN";
  case CUFFT_ALLOC_FAILED:
    return "CUFFT_ALLOC_FAILED";
  case CUFFT_INVALID_TYPE:
    return "CUFFT_INVALID_TYPE";
  case CUFFT_INVALID_VALUE:
    return "CUFFT_INVALID_VALUE";
  case CUFFT_INTERNAL_ERROR:
    return "CUFFT_INTERNAL_ERROR";
  case CUFFT_EXEC_FAILED:
    return "CUFFT_EXEC_FAILED";
  case CUFFT_SETUP_FAILED:
    return "CUFFT_SETUP_FAILED";
  case CUFFT_INVALID_SIZE:
    return "CUFFT_INVALID_SIZE";
  case CUFFT_UNALIGNED_DATA:
    return "CUFFT_UNALIGNED_DATA";
  case CUFFT_INVALID_DEVICE:
    return "CUFFT_INVALID_DEVICE";
  case CUFFT_NO_WORKSPACE:
    return "CUFFT_NO_WORKSPACE";
  case CUFFT_NOT_IMPLEMENTED:
    return "CUFFT_NOT_IMPLEMENTED";
  case CUFFT_NOT_SUPPORTED:
    return "CUFFT_NOT_SUPPORTED";
  case CUFFT_MISSING_DEPENDENCY:
    return "CUFFT_MISSING_DEPENDENCY";
  case CUFFT_NVRTC_FAILURE:
    return "CUFFT_NVRTC_FAILURE";
  case CUFFT_NVJITLINK_FAILURE:
    return "CUFFT_NVJITLINK_FAILURE";
  case CUFFT_NVSHMEM_FAILURE:
    return "CUFFT_NVSHMEM_FAILURE";
  default:
    return "Unknown";
  }
}

void fft::gpu::printCudaError(std::string msg, cudaError_t err)
{
  std::cerr << " *** " << msg << ": "
            << cudaGetErrorString(err)
            << " ***" << std::endl;
}

void fft::gpu::printCudaError(std::string msg, cufftResult err)
{
  std::cerr << " *** " << msg << ": "
            << cufftGetErrorString(err)
            << " ***" << std::endl;
}
