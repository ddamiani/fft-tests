#ifndef _FFT_UTILS_OPTIONPARSER_HH_
#define _FFT_UTILS_OPTIONPARSER_HH_

#include<fft/utils/Options.hh>
#include<ostream>

namespace fft {
  class OptionParser {
  public:
    OptionParser(const char* description);
    virtual ~OptionParser();

    bool ready() const;
    int exit_code() const;
    bool verbose() const;
    DimensionSize ndims() const;
    Dimensions dimensions() const;
    unsigned int iterations() const;
    unsigned int batches() const;
    unsigned int parallelization() const;
    unsigned int maxprint() const;
    Algorithm type() const;

    bool parse(int argc, char *argv[]);

    friend std::ostream& operator<<(std::ostream& out, const OptionParser& opt);

  private:
    const char*   _description;
    bool          _ready;
    int           _exit_code;
    bool          _verbose;
    Dimensions    _dimensions;
    unsigned int  _iterations;
    unsigned int  _batches;
    unsigned int  _parallelization;
    unsigned int  _maxprint;
    Algorithm     _type;
  };
}

#endif
