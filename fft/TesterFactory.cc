#include "TesterFactory.hh"
#include "cpu/fftwTester.hh"
#include "gpu/cufftTester.hh"
#include <iostream>

using namespace fft;
using parser::Algorithm;
using parser::AlgoMap;
using parser::AlgoName;

Tester* TesterFactory::create(Algorithm type)
{
  try {
    switch (type) {
    case Algorithm::fftw:
      return new cpu::fftwTester(AlgoName(type));
    case Algorithm::cuFFT:
      return new gpu::cufftTester(AlgoName(type));
    default:
      return nullptr;
    }
  } catch (const std::out_of_range& oor) {
    std::cerr << "Out of range error on AlgoMap:" << oor.what() << std::endl;
    return nullptr;
  }
}
