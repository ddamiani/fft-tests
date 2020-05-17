#include "OptionParser.hh"
#include "CLI/CLI.hpp"

using namespace fft::parser;

OptionParser::OptionParser(const char* description) :
  _description(description),
  _ready(false),
  _exit_code(0),
  _dimensions(),
  _iterations(1),
  _maxprint(0),
  _type(Algorithm::fftw)
{}

OptionParser::~OptionParser()
{}

bool OptionParser::ready() const { return _ready; }

int OptionParser::exit_code() const { return _exit_code; }

DimensionSize OptionParser::ndims() const { return _dimensions.size(); }

Dimensions OptionParser::dimensions() const { return _dimensions; }

unsigned long OptionParser::iterations() const { return _iterations; }

unsigned long OptionParser::maxprint() const { return _maxprint; }

Algorithm OptionParser::type() const { return _type; }


bool OptionParser::parse(int argc, char *argv[])
{
  CLI::App app{_description};

  /* options */
  app.add_option("-i,--iterations", _iterations, "the number of iterations of the fft to run", true);
  app.add_option("-p,--maxprint", _maxprint, "the maximum number of points of the output to print", true);
  app.add_option("-t,--type", _type, "the type of the fft to run (e.g. fftw, cuFFT, etc)", true)
    ->transform(CLI::CheckedTransformer(AlgoMap));
  /* positionals */
  app.add_option("npoints", _dimensions, "the number of points for each dimension as list")
    ->required()
    ->expected(1,3);

  try {
    app.parse(argc, argv);
    _ready = true;
  } catch (const CLI::ParseError &e) {
    _exit_code = app.exit(e);
    _ready = false;
  }

  return _ready;
}

namespace fft {
  namespace parser {
    std::ostream& operator<<(std::ostream& out, const OptionParser& opt)
    {
      out << " *** Parsed the following command line options:" << std::endl
          << "   Algorithm:  " << opt._type << std::endl
          << "   Iterations: " << opt._iterations << std::endl
          << "   Dimensions: " << opt._dimensions.size() << std::endl
          << "   Data Shape:";

      for (auto dim: opt._dimensions) {
        out << " " << dim;
      }

      return out << std::endl << "   Max Prints: "
                 << opt._maxprint << std::endl;
    }
  }
}
