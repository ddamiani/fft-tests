#ifndef _FFT_UTILS_OPTIONPARSER_HH_
#define _FFT_UTILS_OPTIONPARSER_HH_

#include<fft/utils/Options.hh>
#include<ostream>
#include<vector>
#include<string>
#include<set>

namespace fft {
  namespace parser {
    typedef std::vector<unsigned long> Dimensions;
    typedef std::vector<unsigned long>::size_type DimensionSize;

    class OptionParser {
    public:
      OptionParser(const char* description);
      virtual ~OptionParser();

      bool ready() const;
      int exit_code() const;
      DimensionSize ndims() const;
      Dimensions dimensions() const;
      unsigned long iterations() const;
      unsigned long maxprint() const;
      Algorithm type() const;

      bool parse(int argc, char *argv[]);

      friend std::ostream& operator<<(std::ostream& out, const OptionParser& opt);

    private:
      const char*   _description;
      bool          _ready;
      int           _exit_code;
      Dimensions    _dimensions;
      unsigned long _iterations;
      unsigned long _maxprint;
      Algorithm     _type;
    };
  }
}

#endif
