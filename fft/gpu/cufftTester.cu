#include "cufftTester.hh"
#include "CudaErrors.hh"
#include "fft/utils/Units.hh"
#include "fft/utils/Signals.hh"
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

using namespace fft::gpu;
using namespace fft::math;

constexpr int MAX_GPUS = 16;

cufftBaseTester::cufftBaseTester(const std::string& name,
                                 unsigned int parallelization,
                                 unsigned int flags,
                                 bool verbose) :
  Tester(name, parallelization, flags, verbose),
  _plan(0),
  _signal(nullptr),
  _result(nullptr),
  _devices()
{
  int ndevices;
  cudaError_t err;
  cudaDeviceProp prop;
  if ((err = cudaGetDeviceCount(&ndevices)) != cudaSuccess) {
    printCudaError("CUDAError: Problem getting device count", err);
  } else {
    for (int device=0; device<std::min(ndevices, MAX_GPUS); ++device) {
      if ((err = cudaGetDeviceProperties(&prop, device)) != cudaSuccess) {
        printCudaError("CUDAError: Problem getting device properties", err);
      } else {
        _devices.push_back(device);
        if (this->verbose()) {
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
    }
  }
}

cufftBaseTester::~cufftBaseTester()
{
  if (_plan) {
    cufftDestroy(_plan);
    _plan = 0;
  }
  if (_signal) {
    free(_signal);
    _signal = nullptr;
  }
  if (_result) {
    free(_result);
    _result = nullptr;
  }
}

int cufftBaseTester::ndevices() const
{
  return _devices.size();
}

const int* cufftBaseTester::devices() const
{
  return _devices.data();
}

int* cufftBaseTester::devices()
{
  return _devices.data();
}


bool cufftBaseTester::ready() const
{
  return _plan != 0;
}

bool cufftBaseTester::is_remote() const
{ 
  return true;
}

void cufftBaseTester::destroy_plan()
{
  if (_plan) {
    cufftDestroy(_plan);
    _plan = 0;
  }
}

void cufftBaseTester::display(unsigned int maxprint) const
{
  if (verbose()) {
    unsigned npoints = min(num_points(), maxprint);

    for (unsigned int i = 0; i < npoints; ++i) {
      std::cout << cuCabs(_result[i]) << std::endl;
    }
  }
}

bool cufftBaseTester::_allocate()
{
  // allocate memory and fill the signal array
  if (cufft_alloc()) {
    unsigned int points_per_batch = num_points();
    unsigned int num_batches = batches();

    for (unsigned int b = 0; b < num_batches; ++b) {
      for (unsigned int i = points_per_batch * b; i < points_per_batch * (b + 1); ++i) {
        signal(_signal[i].x, _signal[i].y, (double)i / (double)points_per_batch);
      }
    }
    return true;
  } else {
    return false;
  }
}

bool cufftBaseTester::cufft_alloc()
{
  if (cufft_device_alloc()) {
    return cufft_host_alloc();
  } else {
    return false;
  }
}

bool cufftBaseTester::cufft_host_alloc()
{
  size_t size = num_points() * sizeof(cufftDoubleComplex) * batches();
  // free the signal and result if they already exist
  if (_signal) free(_signal);
  if (_result) free(_result);

  // allocate memory on the host
  _signal = reinterpret_cast<cufftDoubleComplex*>(malloc(size));
  _result = reinterpret_cast<cufftDoubleComplex*>(malloc(size));

  return _signal && _result;
}

cufftTester::cufftTester(const std::string& name,
                         unsigned int flags,
                         bool verbose) :
  cufftBaseTester(name, 1, flags, verbose),
  _dev_data(nullptr)
{}

cufftTester::~cufftTester()
{
  if (_dev_data) {
    cudaFree(_dev_data);
    _dev_data = nullptr;
  }
}

bool cufftTester::send_data() {
  size_t size = num_points() * sizeof(cufftDoubleComplex);
  cudaError_t err = cudaMemcpy(_dev_data, _signal, size, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    printCudaError("CUDAError: Problem copying data to device", err);
    return false;
  } else {
    return true;
  }
}

bool cufftTester::execute()
{
  cufftResult ffterr = cufftExecZ2Z(_plan, _dev_data, _dev_data, CUFFT_FORWARD);
  if (ffterr != CUFFT_SUCCESS) {
    printCudaError("cuFFT Error: Problem executing the plan", ffterr);
    return false;
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
  size_t size = num_points() * sizeof(cufftDoubleComplex);
  cudaError_t err = cudaMemcpy(_result, _dev_data, size, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    printCudaError("CUDAError: Problem copying data to host", err);
    return false;
  } else {
    return true;
  }
}

bool cufftTester::_create_plan()
{
  size_t workSize = 0;
  unsigned int npoints = num_points();

  // create the plan
  cufftResult err = cufftPlanMany(&_plan, rank(), shape(),
                                  NULL, 1, npoints,
                                  NULL, 1, npoints,
                                  CUFFT_Z2Z, batches());
  if (err != CUFFT_SUCCESS) {
    printCudaError("cuFFT Error: Unable to create plan", err);
    return false;
  } else if ((err = cufftGetSize(_plan, &workSize)) != CUFFT_SUCCESS) {
    printCudaError("cuFFT Error: Unable to get plan work size", err);
    return false;
  } else {
    if (verbose()) {
      Value<double> workSizeConv = convert_bytes<double>(workSize);
      std::cout << " *** Created plan with work area size (" << workSizeConv.unit
                << "): " <<  workSizeConv.value << " ***" << std::endl;
    }
    return true;
  }
}

bool cufftTester::cufft_device_alloc()
{
  cudaError_t err;
  size_t size = num_points() * sizeof(cufftDoubleComplex) * batches();

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
    return true;
  }
}

cufftXtTester::cufftXtTester(const std::string& name,
                             unsigned int parallelization,
                             unsigned int flags,
                             bool verbose) :
  cufftBaseTester(name, parallelization, flags, verbose),
  _dev_data(nullptr)
{}

cufftXtTester::~cufftXtTester()
{
  if (_dev_data) {
    cufftXtFree(_dev_data);
    _dev_data = nullptr;
  }
}

bool cufftXtTester::send_data() {
  size_t size = num_points() * sizeof(cufftDoubleComplex);
  cufftResult err = cufftXtMemcpy(_plan, _dev_data, _signal, CUFFT_COPY_HOST_TO_DEVICE);
  if (err != CUFFT_SUCCESS) {
    printCudaError("cuFFT Error: Problem copying data to device", err);
    return false;
  } else {
    return true;
  }
}

bool cufftXtTester::execute()
{
  cufftResult ffterr = cufftXtExecDescriptorZ2Z(_plan, _dev_data, _dev_data, CUFFT_FORWARD);
  if (ffterr != CUFFT_SUCCESS) {
    printCudaError("cuFFT Error: Problem executing the plan", ffterr);
    return false;
  }

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printCudaError("CUDAError: Problem synchronizing device", err);
    return false;
  } else {
    return true;
  }
}

bool cufftXtTester::recv_data()
{
  size_t size = num_points() * sizeof(cufftDoubleComplex);
  cufftResult err = cufftXtMemcpy(_plan, _result, _dev_data, CUFFT_COPY_HOST_TO_DEVICE);
  if (err != CUFFT_SUCCESS) {
    printCudaError("cuFFT Error: Problem copying data to host", err);
    return false;
  } else {
    return true;
  }
}

bool cufftXtTester::_set_num_gpu()
{
  int requested_dev = parallelization();

  // check if enough gpus are available
  if (requested_dev > ndevices()) {
    std::cerr << " *** Requested number of GPUs (" << requested_dev
              << ") is higher than number available ("  << ndevices()
              << ")! ***" << std::endl;
    return false;
  }

  cufftResult err = cufftXtSetGPUs(_plan, requested_dev, devices());
  if (err != CUFFT_SUCCESS) {
    printCudaError("cuFFT Error: Unable to set the number of GPUs", err);
    return false;
  } else {
    return true;
  }
}

bool cufftXtTester::_alloc_needs_plan() const
{
  return true;
}

bool cufftXtTester::_create_plan()
{
  size_t workSize = 0;
  unsigned int npoints = num_points();

  // create and empty plan
  cufftResult err = cufftCreate(&_plan);
  if (err != CUFFT_SUCCESS) {
    printCudaError("cuFFT Error: Unable to create empty plan", err);
    return false;
  } else if (!_set_num_gpu()) {
    std::cerr << " *** Failed to set the "
              << "number of GPUs used for the FFT to "
              << parallelization() << "! ***" << std::endl;
    return false;
  } else {
    // create the plan
    cufftResult err = cufftMakePlanMany(_plan, rank(), shape(),
                                        NULL, 1, npoints,
                                        NULL, 1, npoints,
                                        CUFFT_Z2Z, batches(), &workSize);
    if (err != CUFFT_SUCCESS) {
      printCudaError("cuFFT Error: Unable to create plan", err);
      return false;
    } else {
      if (verbose()) {
        Value<double> workSizeConv = convert_bytes<double>(workSize);
        std::cout << " *** Created plan with work area size (" << workSizeConv.unit
                  << "): " <<  workSizeConv.value << " ***" << std::endl;
      }
      return true;
    }
  }
}

bool cufftXtTester::cufft_device_alloc()
{
  cufftResult err;
  size_t size = num_points() * sizeof(cufftDoubleComplex) * batches();

  // free the device memory if needed
  if (_dev_data) {
    if ((err = cufftXtFree(_dev_data)) != CUFFT_SUCCESS) {
      printCudaError("cuFFT Error: failure freeing device memory", err);
      return false;
    } else {
      _dev_data = nullptr;
    }
  }

  // allocate memory on the device
  if ((err = cufftXtMalloc(_plan, &_dev_data, CUFFT_XT_FORMAT_INPLACE)) != CUFFT_SUCCESS) {
    printCudaError("cuFFT Error: failure allocating device memory", err);
    return false;
  } else {
    return true;
  }
}
