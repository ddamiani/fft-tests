#ifndef _FFT_TESTER_FACTORY_HH_
#define _FFT_TESTER_FACTORY_HH_

#include <fft/Tester.hh>
#include <fft/utils/Options.hh>
#include <memory>

namespace fft {
  class TesterFactory {
  public:
    static std::shared_ptr<Tester> create(Algorithm type,
                                          unsigned int parallelization,
                                          unsigned int flags,
                                          bool verbose=false);
  };
}

#endif
