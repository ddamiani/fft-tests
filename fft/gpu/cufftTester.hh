#ifndef _FFT_GPU_CUFFTTESTER_HH_
#define _FFT_GPU_CUFFTTESTER_HH_

#include <fft/Tester.hh>
#include <cufft.h>

namespace fft {
  namespace gpu {
    class cufftTester : public Tester {
      public:
        cufftTester(const std::string& name);
        virtual ~cufftTester();

        virtual unsigned long num_points() const;
        virtual unsigned int num_dimensions() const;
        virtual bool ready() const;
        virtual bool is_remote() const;

        virtual void create_plan(unsigned long d1);
        virtual void create_plan(unsigned long d1, unsigned long d2);
        virtual void create_plan(unsigned long d1, unsigned long d2, unsigned long d3);
        virtual bool send_data();
        virtual bool execute(unsigned long iterations);
        virtual bool recv_data();

        virtual void display(unsigned long maxprint) const;

      protected:
        bool allocate(unsigned long d1);
        bool allocate(unsigned long d1, unsigned long d2);
        bool allocate(unsigned long d1, unsigned long d2, unsigned long d3);

        bool cufft_alloc();

      protected:
        bool                _ready;
        unsigned int        _ndims;
        unsigned long       _npoints;
        cufftHandle         _plan;
        cufftDoubleComplex* _signal;
        cufftDoubleComplex* _result;
        cufftDoubleComplex* _dev_data;
    };
  }
}

#endif
