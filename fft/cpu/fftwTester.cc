#include "fftwTester.hh"
#include "fft/utils/Signals.hh"
#include <cmath>
#include <algorithm>
#include <iostream>

constexpr int REAL = 0;
constexpr int IMAG = 1;

using namespace fft::cpu;
using namespace fft::math;

fftwTester::fftwTester(const std::string& name,
                       unsigned int flags,
                       bool verbose) :
  Tester(name, flags, 1, verbose),
  _plan(nullptr),
  _signal(nullptr),
  _result(nullptr)
{}

fftwTester::~fftwTester()
{
  if (_plan) {
    fftw_destroy_plan(_plan);
    _plan = nullptr;
  }
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

bool fftwTester::ready() const
{
  return _plan != nullptr;
}

bool fftwTester::_create_plan()
{
  unsigned int npoints = num_points();

  // create the plan
  _plan = fftw_plan_many_dft(rank(), shape(), batches(),
                             _signal, NULL, 1, npoints,
                             _result, NULL, 1, npoints,
                             FFTW_FORWARD, FFTW_ESTIMATE);

  return _plan != nullptr;
}

void fftwTester::destroy_plan()
{
  if (_plan) {
    fftw_destroy_plan(_plan);
    _plan = nullptr;
  }
}

bool fftwTester::execute()
{
  fftw_execute(_plan);

  return true;
}

void fftwTester::display(unsigned int maxprint) const
{
  if (verbose()) {
    unsigned int npoints = std::min(num_points(), maxprint);

    for (unsigned int i = 0; i < npoints; ++i) {
      std::cout << std::hypot(_result[i][REAL], _result[i][IMAG]) << std::endl;
    }
  }
}

bool fftwTester::_allocate()
{
  // get size info
  unsigned int points_per_batch = num_points();
  unsigned int num_batches = batches();
  size_t total_points = points_per_batch * num_batches;

  // free the signal and result if they already exist
  if (_signal) fftw_free(_signal);
  if (_result) fftw_free(_result);

  // allocate memory
  _signal = fftw_alloc_complex(total_points);
  _result = fftw_alloc_complex(total_points);

  for (unsigned int b = 0; b < num_batches; ++b) {
    for (unsigned int i = points_per_batch * b; i < points_per_batch * (b + 1); ++i) {
      signal(_signal[i][REAL], _signal[i][IMAG], (double)i / (double)points_per_batch);
    }
  }

  return true;
}
