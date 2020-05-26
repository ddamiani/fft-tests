#ifndef _FFT_GPU_CUFFTTESTER_HH_
#define _FFT_GPU_CUFFTTESTER_HH_

#include <fft/Tester.hh>
#include <cufftXt.h>
#include <vector>

namespace fft {
  namespace gpu {
    class cufftBaseTester : public Tester {
    public:
      cufftBaseTester(const std::string& name,
                      unsigned int parallelization,
                      unsigned int flags,
                      bool verbose=false);
      virtual ~cufftBaseTester();

      virtual int ndevices() const;
      virtual const int* devices() const;
      virtual int* devices();
      virtual bool ready() const;
      virtual bool is_remote() const;
      virtual void destroy_plan();
      virtual void display(unsigned int maxprint) const;

    protected:
      virtual bool _allocate();
      virtual bool cufft_alloc();
      virtual bool cufft_host_alloc();
      virtual bool cufft_device_alloc() = 0;

    protected:
      cufftHandle         _plan;
      cufftDoubleComplex* _signal;
      cufftDoubleComplex* _result;
    private:
      std::vector<int>    _devices;
    };

    class cufftTester : public cufftBaseTester {
    public:
      cufftTester(const std::string& name,
                  unsigned int flags,
                  bool verbose=false);
      virtual ~cufftTester();

      virtual bool send_data();
      virtual bool execute();
      virtual bool recv_data();

    protected:
      virtual bool _create_plan();
      virtual bool cufft_device_alloc();

    protected:
      cufftDoubleComplex* _dev_data;
    };

    class cufftXtTester : public cufftBaseTester {
    public:
      cufftXtTester(const std::string& name,
                    unsigned int parallelization,
                    unsigned int flags,
                    bool verbose=false);
      virtual ~cufftXtTester();

      virtual bool send_data();
      virtual bool execute();
      virtual bool recv_data();

    protected:
      virtual bool _set_num_gpu();
      virtual bool _alloc_needs_plan() const;
      virtual bool _create_plan();
      virtual bool cufft_device_alloc();

    protected:
      cudaLibXtDesc* _dev_data;
    };
  }
}

#endif
