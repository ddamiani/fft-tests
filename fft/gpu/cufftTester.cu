#include "cufftTester.hh"
#include "CudaErrors.hh"
#include "fft/utils/Units.hh"
#include "fft/utils/Signals.hh"
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

using namespace fft::gpu;
using namespace fft::math;

cufftTester::cufftTester(const std::string& name) :
  Tester(name),
  _ready(false),
  _ndims(0),
  _npoints(0),
  _signal(nullptr),
  _result(nullptr),
  _dev_data(nullptr)
{
  int device;
  cudaError_t err;
  cudaDeviceProp prop;
  if ((err = cudaGetDevice(&device)) != cudaSuccess) {
    printCudaError("CUDAError: Problem get current device", err);    
  } else if ((err = cudaGetDeviceProperties(&prop, device)) != cudaSuccess) {
    printCudaError("CUDAError: Problem getting device properties", err);
  } else {
    std::cout << " *** GPU Device Properties:" << std::endl
              << "   Device Number:                " << device << std::endl
              << "   Device Name:                  " << prop.name << std::endl
              << "   Compute Capability:           " << prop.major << "." << prop.minor << std::endl
              << "   Memory Clock Rate (KHz):      " << prop.memoryClockRate << std::endl
              << "   Memory Bus Width (bits):      " << prop.memoryBusWidth << std::endl
              << "   Peak Memory Bandwidth (GB/s): " 
              << 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6 << std::endl
              << "   Total Constant Memory (KB):   " << prop.totalConstMem/1024.0 << std::endl
              << "   Total Global Memory (GB):     " << prop.totalGlobalMem/pow(1024, 3)
              << std::endl;
  }
}

cufftTester::~cufftTester()
{
  _ready = false;
  cufftDestroy(_plan);
  if (_signal) {
    free(_signal);
    _signal = nullptr;
  }
  if (_result) {
    free(_result);
    _result = nullptr;
  }
  if (_dev_data) {
    cudaFree(_dev_data);
    _dev_data = nullptr;
  }
}

unsigned long cufftTester::num_points() const
{
  return _npoints;
}

unsigned int cufftTester::num_dimensions() const
{
  return _ndims;
}

bool cufftTester::ready() const
{
  return _ready;
}

bool cufftTester::is_remote() const
{ 
  return true;
}

void cufftTester::create_plan(unsigned long d1)
{
  size_t workSize = 0;

  _ready = false;
  _ndims = 0;
  // allocate memory
  if (allocate(d1)) {
    // create the plan
    cufftResult err = cufftPlan1d(&_plan, d1, CUFFT_Z2Z, 1);
    if (err != CUFFT_SUCCESS) {
      printCudaError("cuFFT Error: Unable to create plan", err);
    } else {
      // set the dims
      _ndims = 1;
      _ready = true;
      if ((err = cufftGetSize(_plan, &workSize)) != CUFFT_SUCCESS) {
        printCudaError("cuFFT Error: Unable to get plan work size", err);
      } else {
        Value<double> workSizeConv = convert_bytes<double>(workSize);
        std::cout << "Created plan with work area size (" << workSizeConv.unit
                  << "): " <<  workSizeConv.value << std::endl;
      }
    }
  }
}

void cufftTester::create_plan(unsigned long d1, unsigned long d2)
{
  size_t workSize = 0;

  _ready = false;
  _ndims = 0;
  // allocate memory
  if (allocate(d1, d2)) {
    // create the plan
    cufftResult err = cufftPlan2d(&_plan, d1, d2, CUFFT_Z2Z);
    if (err != CUFFT_SUCCESS) {
      printCudaError("cuFFT Error: Unable to create plan", err);
    } else {
      // set the dims
      _ndims = 2;
      _ready = true;
      if ((err = cufftGetSize(_plan, &workSize)) != CUFFT_SUCCESS) {
        printCudaError("cuFFT Error: Unable to get plan work size", err);
      } else {
        Value<double> workSizeConv = convert_bytes<double>(workSize);
        std::cout << "Created plan with work area size (" << workSizeConv.unit
                  << "): " <<  workSizeConv.value << std::endl;
      }
    }
  }
}

void cufftTester::create_plan(unsigned long d1, unsigned long d2, unsigned long d3)
{
  size_t workSize = 0;

  _ready = false;
  _ndims = 0;
  // allocate memory
  if (allocate(d1, d2, d3)) {
    // create the plan
    cufftResult err = cufftPlan3d(&_plan, d1, d2, d3, CUFFT_Z2Z);
    if (err != CUFFT_SUCCESS) {
      printCudaError("cuFFT Error: Unable to create plan", err);
    } else {
      // set the dims
      _ndims = 3;
      _ready = true;
      if ((err = cufftGetSize(_plan, &workSize)) != CUFFT_SUCCESS) {
        printCudaError("cuFFT Error: Unable to get plan work size", err);
      } else {
        Value<double> workSizeConv = convert_bytes<double>(workSize);
        std::cout << "Created plan with work area size (" << workSizeConv.unit
                  << "): " <<  workSizeConv.value << std::endl;
      }
    }
  }
}

bool cufftTester::send_data() {
  size_t size = _npoints * sizeof(cufftDoubleComplex);
  cudaError_t err = cudaMemcpy(_dev_data, _signal, size, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    printCudaError("CUDAError: Problem copying data to device", err);
    return false;
  } else {
    return true;
  }
}

bool cufftTester::execute(unsigned long iterations)
{
  for (unsigned long i=0; i<iterations; ++i) {
    cufftResult ffterr = cufftExecZ2Z(_plan, _dev_data, _dev_data, CUFFT_FORWARD);
    if (ffterr != CUFFT_SUCCESS) {
      printCudaError("cuFFT Error: Problem executing the plan", ffterr);
      return false;
    }
  }

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printCudaError("CUDAError: Problem synchronizing device", err);
    return false;
  } else {
    return true;
  }
}

bool cufftTester::recv_data()
{
  size_t size = _npoints * sizeof(cufftDoubleComplex);
  cudaError_t err = cudaMemcpy(_result, _dev_data, size, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    printCudaError("CUDAError: Problem copying data to host", err);
    return false;
  } else {
    return true;
  }
}

void cufftTester::display(unsigned long maxprint) const
{
  unsigned long num_points = min(_npoints, maxprint);

  for (unsigned long i = 0; i < num_points; ++i) {
    std::cout << cuCabs(_result[i]) << std::endl;
  }
}

bool cufftTester::allocate(unsigned long d1)
{
  _npoints =  d1;

  // allocate memory and fill the signal array
  if (cufft_alloc()) {
    for (unsigned long i = 0; i < _npoints; ++i) {
      signal(_signal[i].x, _signal[i].y, (double)i / (double)_npoints);
    }
    return true;
  } else {
    return false;
  }
}

bool cufftTester::allocate(unsigned long d1, unsigned long d2)
{
  return allocate(d1 * d2);
}

bool cufftTester::allocate(unsigned long d1, unsigned long d2, unsigned long d3)
{
  return allocate(d1 * d2 * d3);
}

bool cufftTester::cufft_alloc()
{
  cudaError_t err;
  size_t size = _npoints * sizeof(cufftDoubleComplex);

  // free the device memory if needed
  if (_dev_data) {
    if ((err = cudaFree(_dev_data)) != cudaSuccess) {
      printCudaError("CUDAError: failure freeing device memory", err);
      return false;
    } else {
      _dev_data = nullptr;
    }
  }

  // allocate memory on the device
  if ((err = cudaMalloc((void**)&_dev_data, size)) != cudaSuccess) {
    printCudaError("CUDAError: failure allocating device memory", err);
    return false;
  } else {
    // free the signal and result if they already exist
    if (_signal) free(_signal);
    if (_result) free(_result);

    // allocate memory on the host
    _signal = reinterpret_cast<cufftDoubleComplex*>(malloc(size));
    _result = reinterpret_cast<cufftDoubleComplex*>(malloc(size));

    return true;
  }
}
