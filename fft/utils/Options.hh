#ifndef _FFT_UTILS_OPTIONFLAGS_HH_
#define _FFT_UTILS_OPTIONFLAGS_HH_

#include <map>
#include <string>

namespace fft {
  namespace parser {
    enum class Flag : unsigned {
      PinnedMemory = 0,
      PinnedMemoryMapped = 1,
    };

    enum class Algorithm : unsigned {
      fftw,
      cuFFT,
      cuFFTXt,
    };

    std::string AlgoName(Algorithm algo) {
      switch (algo) {
      case Algorithm::fftw:
        return "fftw";
      case Algorithm::cuFFT:
        return "cuFFT";
      case Algorithm::cuFFTXt:
        return "cuFFTXt";
      default:
        return "";
      }
    }

    static const std::map<std::string, Algorithm> AlgoMap{
      {"fftw", Algorithm::fftw},
      {"cuFFT", Algorithm::cuFFT},
      {"cuFFTXt", Algorithm::cuFFTXt}
    };

    std::ostream& operator<<(std::ostream& out, const Algorithm& algo)
    {
      return out << AlgoName(algo);
    }
  }
}

#endif
