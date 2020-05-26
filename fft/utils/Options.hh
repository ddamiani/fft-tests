#ifndef _FFT_UTILS_OPTIONFLAGS_HH_
#define _FFT_UTILS_OPTIONFLAGS_HH_

#include <map>
#include <vector>
#include <string>

namespace fft {
  typedef std::vector<int> Dimensions;
  typedef std::vector<int>::size_type DimensionSize;

  enum class Flag : unsigned {
    PinnedMemory = 0,
    PinnedMemoryMapped = 1,
  };

  enum class Algorithm : unsigned {
    fftw,
    cuFFT,
  };

  std::string AlgoName(Algorithm algo) {
    switch (algo) {
    case Algorithm::fftw:
      return "fftw";
    case Algorithm::cuFFT:
      return "cuFFT";
    default:
      return "";
    }
  }

  static const std::map<std::string, Algorithm> AlgoMap{
    {"fftw", Algorithm::fftw},
    {"cuFFT", Algorithm::cuFFT},
  };

  std::ostream& operator<<(std::ostream& out, const Algorithm& algo)
  {
    return out << AlgoName(algo);
  }
}

#endif
