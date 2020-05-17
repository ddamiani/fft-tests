#ifndef _FFT_TESTER_FACTORY_HH_
#define _FFT_TESTER_FACTORY_HH_

#include <fft/Tester.hh>
#include <fft/utils/Options.hh>

namespace fft {
  class TesterFactory {
  public:
    static Tester* create(parser::Algorithm type);
  };
}

#endif
