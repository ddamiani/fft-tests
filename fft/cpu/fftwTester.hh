#ifndef _FFT_CPU_FFTWTESTER_HH_
#define _FFT_CPU_FFTWTESTER_HH_

#include <fft/Tester.hh>
#include <fftw3.h>

namespace fft {
  namespace cpu {
    class fftwTester : public Tester {
      public:
        fftwTester(const std::string& name,
                   unsigned int flags,
                   bool verbose=false);
        virtual ~fftwTester();

        virtual bool ready() const;
        virtual void destroy_plan();
        virtual bool execute();
        virtual void display(unsigned int maxprint) const;

      protected:
        virtual bool _allocate();
        virtual bool _create_plan();

      protected:
        fftw_plan     _plan;
        fftw_complex* _signal;
        fftw_complex* _result;
    };
  }
}

#endif
