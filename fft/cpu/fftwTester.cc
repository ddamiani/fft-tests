#include "fftwTester.hh"
#include "fft/utils/Signals.hh"
#include <cmath>
#include <algorithm>
#include <iostream>

constexpr int REAL = 0;
constexpr int IMAG = 1;

using namespace fft::cpu;
using namespace fft::math;

fftwTester::fftwTester(const std::string& name) :
  Tester(name),
  _ready(false),
  _ndims(0),
  _npoints(0),
  _signal(nullptr),
  _result(nullptr)
{}

fftwTester::~fftwTester()
{
  _ready = false;
  fftw_destroy_plan(_plan);
  fftw_cleanup();
  if (_signal) {
    fftw_free(_signal);
    _signal = nullptr;
  }
  if (_result) {
    fftw_free(_result);
    _result = nullptr;
  }
}

unsigned long fftwTester::num_points() const
{
  return _npoints;
}

unsigned int fftwTester::num_dimensions() const
{
  return _ndims;
}

bool fftwTester::ready() const
{
  return _ready;
}

void fftwTester::create_plan(unsigned long d1)
{
  _ready = false;
  // allocate memory
  allocate(d1);
  // create the plan
  _plan = fftw_plan_dft_1d(d1, _signal, _result, FFTW_FORWARD, FFTW_ESTIMATE);
  // set the dims
  _ndims = 1;
  _ready = true;
}

void fftwTester::create_plan(unsigned long d1, unsigned long d2)
{
  _ready = false;
  // allocate memory
  allocate(d1, d2);
  // create the plan
  _plan = fftw_plan_dft_2d(d1, d2, _signal, _result, FFTW_FORWARD, FFTW_ESTIMATE);
  // set the dims
  _ndims = 2;
  _ready = true;
}

void fftwTester::create_plan(unsigned long d1, unsigned long d2, unsigned long d3)
{
  _ready = false;
  // allocate memory
  allocate(d1, d2, d3);
  // create the plan
  _plan = fftw_plan_dft_3d(d1, d2, d3, _signal, _result, FFTW_FORWARD, FFTW_ESTIMATE);
  // set the dims
  _ndims = 3;
  _ready = true;
}

bool fftwTester::execute(unsigned long iterations)
{
  for (unsigned long i=0; i<iterations; ++i) {
    fftw_execute(_plan);
  }

  return true;
}

void fftwTester::display(unsigned long maxprint) const
{
  unsigned long num_points = std::min(_npoints, maxprint);

  for (unsigned long i = 0; i < num_points; ++i) {
    std::cout << std::hypot(_result[i][REAL], _result[i][IMAG]) << std::endl;
  }
}

void fftwTester::allocate(unsigned long d1)
{
  _npoints =  d1;

  // free the signal and result if they already exist
  if (_signal) fftw_free(_signal);
  if (_result) fftw_free(_result);

  // allocate memory
  _signal = fftw_alloc_complex(_npoints);
  _result = fftw_alloc_complex(_npoints);

  for (unsigned long i = 0; i < _npoints; ++i) {
    signal(_signal[i][REAL], _signal[i][IMAG], (double)i / (double)_npoints);
  }
}

void fftwTester::allocate(unsigned long d1, unsigned long d2)
{
  allocate(d1 * d2);
}

void fftwTester::allocate(unsigned long d1, unsigned long d2, unsigned long d3)
{
  allocate(d1 * d2 * d3);
}
