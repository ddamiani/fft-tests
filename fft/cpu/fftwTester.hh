#ifndef _FFT_CPU_FFTWTESTER_HH_
#define _FFT_CPU_FFTWTESTER_HH_

#include <fft/Tester.hh>
#include <fftw3.h>

namespace fft {
  namespace cpu {
    class fftwTester : public Tester {
      public:
        fftwTester(const std::string& name);
        virtual ~fftwTester();

        virtual unsigned long num_points() const;
        virtual unsigned int num_dimensions() const;
        virtual bool ready() const;

        virtual void create_plan(unsigned long d1);
        virtual void create_plan(unsigned long d1, unsigned long d2);
        virtual void create_plan(unsigned long d1, unsigned long d2, unsigned long d3);
        virtual bool execute(unsigned long iterations);

        virtual void display(unsigned long maxprint) const;

      protected:
        void allocate(unsigned long d1);
        void allocate(unsigned long d1, unsigned long d2);
        void allocate(unsigned long d1, unsigned long d2, unsigned long d3);

      protected:
        bool          _ready;
        unsigned int  _ndims;
        unsigned long _npoints;
        fftw_plan     _plan;
        fftw_complex* _signal;
        fftw_complex* _result;
    };
  }
}

#endif
