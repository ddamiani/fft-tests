#include "TesterFactory.hh"
#include "cpu/fftwTester.hh"
#include "gpu/cufftTester.hh"
#include <iostream>

using namespace fft;

Tester* TesterFactory::create(Algorithm type,
                              unsigned int parallelization,
                              unsigned int flags,
                              bool verbose)
{
  try {
    switch (type) {
    case Algorithm::fftw:
      return new cpu::fftwTester(AlgoName(type), flags, verbose);
    case Algorithm::cuFFT:
      if (parallelization > 1)
        return new gpu::cufftXtTester(AlgoName(type), parallelization, flags, verbose);
      else
        return new gpu::cufftTester(AlgoName(type), flags, verbose);
    default:
      return nullptr;
    }
  } catch (const std::out_of_range& oor) {
    std::cerr << "Out of range error on AlgoMap:" << oor.what() << std::endl;
    return nullptr;
  }
}
