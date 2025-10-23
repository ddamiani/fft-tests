#include "OptionParser.hh"
#include "CLI/CLI.hpp"

using namespace fft;

OptionParser::OptionParser(const char* description) :
  _description(description),
  _ready(false),
  _exit_code(0),
  _verbose(false),
  _dimensions(),
  _iterations(1),
  _batches(1),
  _parallelization(1),
  _maxprint(0),
  _type(Algorithm::fftw)
{}

OptionParser::~OptionParser()
{}

bool OptionParser::ready() const { return _ready; }

int OptionParser::exit_code() const { return _exit_code; }

DimensionSize OptionParser::ndims() const { return _dimensions.size(); }

Dimensions OptionParser::dimensions() const { return _dimensions; }

bool OptionParser::verbose() const { return _verbose || _iterations==1; }

unsigned int OptionParser::iterations() const { return _iterations; }

unsigned int OptionParser::batches() const { return _batches; }

unsigned int OptionParser::parallelization() const { return _parallelization; }

unsigned int OptionParser::maxprint() const { return _maxprint; }

Algorithm OptionParser::type() const { return _type; }


bool OptionParser::parse(int argc, char *argv[])
{
  CLI::App app{_description};

  /* options */
  app.add_option("-i,--iterations", _iterations, "the number of iterations of the fft to run")
    ->check(CLI::PositiveNumber);
  app.add_option("-b,--batches",  _batches, "the number of batches to run in the fft")
    ->check(CLI::PositiveNumber);
  app.add_option("-m,--maxprint", _maxprint, "the maximum number of points of the output to print");
  app.add_option("-t,--type", _type, "the type of the fft to run (e.g. fftw, cuFFT, etc)")
    ->transform(CLI::CheckedTransformer(AlgoMap));
  app.add_option("-p,--parallel", _parallelization, "the number of parallel tasks to use")
    ->check(CLI::Range(1,16));
  /* positionals */
  app.add_option("npoints", _dimensions, "the number of points for each dimension as list")
    ->required()
    ->expected(1,3);
  /* flags */
  app.add_flag("-v,--verbose", _verbose, "enable verbose output");

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
  std::ostream& operator<<(std::ostream& out, const OptionParser& opt)
  {
    out << " *** Parsed the following command line options:" << std::endl
        << "   Algorithm:  " << opt._type << std::endl
        << "   Iterations: " << opt._iterations << std::endl
        << "   Batches:    " << opt._batches << std::endl
        << "   Dimensions: " << opt._dimensions.size() << std::endl
        << "   Data Shape:";

    for (auto dim: opt._dimensions) {
      out << " " << dim;
    }

    return out << std::endl << "   Max Prints: "
               << opt._maxprint << std::endl;
  }
}
