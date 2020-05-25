#include "Signals.hh"
#define _USE_MATH_DEFINES
#include <cmath>

void fft::math::signal(double& re, double& im, double frac) {
    double theta = frac * M_PI;                                                                                             re = 1.0 * std::cos(10.0 * theta) +0.5 * std::cos(25.0 * theta);                                                        im = 1.0 * std::sin(10.0 * theta) +0.5 * std::sin(25.0 * theta);
}
