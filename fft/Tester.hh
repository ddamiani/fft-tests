#ifndef _FFT_TESTER_HH_
#define _FFT_TESTER_HH_

#include <string>

namespace fft {
  class Tester {
  public:
    Tester(const std::string& name) :
      _name(name) {}

    virtual ~Tester() {}

    std::string name() const { return _name; }

    virtual unsigned long num_points() const = 0;
    virtual unsigned int num_dimensions() const = 0;
    virtual bool ready() const = 0;
    virtual bool is_remote() const { return false; }

    virtual void create_plan(unsigned long d1) = 0;
    virtual void create_plan(unsigned long d1, unsigned long d2) = 0;
    virtual void create_plan(unsigned long d1, unsigned long d2, unsigned long d3) = 0;
    virtual bool send_data() { return true; }
    virtual bool execute(unsigned long iterations) = 0;
    virtual bool recv_data() { return true; }

    virtual void display(unsigned long maxprint) const = 0;

  private:
    std::string _name;
  };
}

#endif
