#include "TesterFactory.hh"
#include "OptionParser.hh"
#include "config.hh"
#include <iostream>
#include <iomanip>
#include <ctime>
#include <vector>

using namespace std;
using namespace fft;

enum RetCodes {
  SUCCESS = 0,
  BAD_ALGO = 1,
  PLAN_FAILURE = 2,
  MEM_COPY_FAILURE = 3,
  EXEC_FAILURE = 4,
};

static void showTiming(const char* msg, clock_t begin, clock_t end)
{
  std::cout << " *** " << msg << ": "
            << std::fixed << std::setprecision(6)
            << (double)(end - begin) / CLOCKS_PER_SEC
            << " s ***" << std::endl;
}

int main(int argc, char *argv[]) {
  clock_t start, finish, begin, end;

  parser::OptionParser opts{FFT_NAME " " FFT_VERSION};
  if (opts.parse(argc, argv)) {
    std::cout << opts;
  } else {
    return opts.exit_code();
  }

  start = clock(); // algo start

  Tester* tester = TesterFactory::create(opts.type());
  if (tester) {
    RetCodes retcode = SUCCESS;
    parser::Dimensions dims = opts.dimensions();
    // create the plan
    begin = clock();
    switch (opts.ndims()) {
    case 1:
      tester->create_plan(dims[0]);
      break;
    case 2:
      tester->create_plan(dims[0], dims[1]);
      break;
    case 3:
      tester->create_plan(dims[0], dims[1], dims[2]);
      break;
    }
    end = clock();
    showTiming("Time to create plan", begin, end);

    // execute the plan if setup succeeded
    if (tester->ready()) {
      bool success = true;

      begin = clock();
      success = tester->send_data();
      end = clock();
      // only show copy time if copy is needed
      if (tester->is_remote()) {
        showTiming("Time to copy data to device", begin, end);
      }

      if (success) {
        begin = clock();
        success = tester->execute(opts.iterations());
        end = clock();
        showTiming("Time to execute plan", begin, end);

        if (success) {
          begin = clock();
          success = tester->recv_data();
          end = clock();
          // only show copy time if copy is needed
          if (tester->is_remote()) {
            showTiming("Time to copy data to host", begin, end);
          }

          if (success) {
            begin = clock();
            tester->display(opts.maxprint());
            end = clock();
            showTiming("Time to display results", begin, end);
          } else {
            std::cerr << " *** Failure copying data to host! ***" << std::endl;
            retcode = MEM_COPY_FAILURE;
          }
        } else {
          std::cerr << " *** Failure encountered executing plan! ***" << std::endl;
          retcode = PLAN_FAILURE;
        }
      } else {
        std::cerr << " *** Failure copying data to device! ***" << std::endl;
        retcode = MEM_COPY_FAILURE;
      }
    } else {
      std::cerr << " *** Failure encountered creating plan! ***" << std::endl;
      retcode = MEM_COPY_FAILURE;
    }

    begin = clock();
    delete tester;
    end = clock();
    showTiming("Time to teardown/cleanup plan", begin, end);

    finish = clock(); // algo end
    showTiming("Total time to complete algorithm", start, finish);

    return retcode;
  } else {
    std::cerr << "Unsupported fft algorithm type: " << opts.type() << std::endl;
    return BAD_ALGO;
  }
}
