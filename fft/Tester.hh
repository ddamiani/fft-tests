#ifndef _FFT_TESTER_HH_
#define _FFT_TESTER_HH_

#include <fft/utils/Options.hh>
#include <vector>
#include <string>

namespace fft {
  typedef std::vector<unsigned int> Shape;

  class Tester {
  public:
    Tester(const std::string& name,
           unsigned int parallelization,
           unsigned int flags,
           bool verbose=false) :
      _name(name),
      _parallelization(parallelization),
      _flags(flags),
      _verbose(verbose),
      _npoints(0),
      _batches(0),
      _dims() {}

    virtual ~Tester() {}

    virtual std::string name() const final { return _name; }
    virtual unsigned int parallelization() const { return _parallelization; }
    virtual unsigned int num_points() const { return _npoints; }
    virtual int rank() const { return static_cast<int>(_dims.size()); }
    virtual Dimensions dimensions() const { return _dims; }
    virtual unsigned int batches() const { return _batches; }
    virtual bool ready() const = 0;
    virtual bool is_remote() const { return false; }
    virtual bool verbose() const final { return _verbose; }

    virtual void create_plan(const Dimensions& dims, unsigned int batches)
    {
      destroy_plan();
      if (_alloc_needs_plan()) {
        if (_create_plan())
          _alloc(dims, batches);
        else
          destroy_plan();
      } else {
        if (_alloc(dims, batches))
          if (!_create_plan())
            destroy_plan();
      }
    }
    virtual void destroy_plan() = 0;

    virtual bool send_data() { return true; }
    virtual bool execute() = 0;
    virtual bool recv_data() { return true; }

    virtual void display(unsigned int maxprint) const = 0;

  protected:
    virtual int* shape() { return _dims.data(); }
    virtual const int* shape() const { return _dims.data(); };
    virtual bool _alloc_needs_plan() const { return false; }
    virtual bool _allocate() = 0;
    virtual bool _create_plan() = 0;

  private:
    virtual bool _alloc(const Dimensions& dims, unsigned int batches)
    {
      _npoints=1;
      for (Dimensions::const_iterator it=dims.begin(); it!=dims.end(); ++it)
        _npoints *= *it;
      _batches = batches;
      _dims = dims;
      return _allocate();
    }

  private:
    std::string   _name;
    unsigned int  _parallelization;
    unsigned int  _flags;
    bool          _verbose;
    unsigned int  _npoints;
    unsigned int  _batches;
    Dimensions    _dims;
  };
}

#endif
