#ifndef _FFT_UTILS_UNITS_HH_
#define _FFT_UTILS_UNITS_HH_

#include <string>

namespace fft {
  namespace math {
    constexpr unsigned int KB = 1024;
    constexpr unsigned int MB = KB * 1024;
    constexpr unsigned int GB = MB * 1024;

    template <typename T>
    struct Value {
      T value;
      std::string unit;
    };

    template <typename T>
    Value<T> convert_bytes(T bytes) {
      if (bytes < KB) {
        return {bytes, "B"};
      } else if (bytes < MB) {
        return {bytes / KB, "KB"};
      } else if (bytes < GB) {
        return {bytes / MB, "MB"};
      } else {
        return {bytes / GB, "GB"};
      }
    }
  } // namespace math
} // namespace fft

#endif
