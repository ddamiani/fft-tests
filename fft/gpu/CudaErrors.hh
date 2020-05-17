#ifndef _FFT_GPU_CUDAERRORS_HH_
#define _FFT_GPU_CUDAERRORS_HH_

#include <cuda_runtime.h>
#include <cufft.h>
#include <string>

namespace fft {
  namespace gpu {
    void printCudaError(std::string msg, cudaError_t err);
    void printCudaError(std::string msg, cufftResult err);
  }
}

#endif
