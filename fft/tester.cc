#include "TesterFactory.hh"
#include "OptionParser.hh"
#include "config.hh"
#include <iostream>
#include <iomanip>
#include <ctime>
#include <vector>

using namespace fft;

enum RetCodes {
  SUCCESS = 0,
  BAD_ALGO = 1,
  PLAN_FAILURE = 2,
  MEM_COPY_FAILURE = 3,
  EXEC_FAILURE = 4,
};

static void showTiming(const char* msg, clock_t begin, clock_t end, bool show=true)
{
  if(show) {
    std::cout << " *** " << msg << ": "
              << std::fixed << std::setprecision(6)
              << (double)(end - begin) / CLOCKS_PER_SEC
              << " s ***" << std::endl;
  }
}

static RetCodes run_fft(Tester* tester, const OptionParser& opts)
{
  clock_t start, finish, begin, end;
  RetCodes retcode = SUCCESS;
  Dimensions dims = opts.dimensions();

  start = clock(); // algo start

  // create the plan
  begin = clock();
  tester->create_plan(dims, opts.batches());
  end = clock();
  showTiming("Time to create plan", begin, end, opts.verbose());

  // execute the plan if setup succeeded
  if (tester->ready()) {
    bool success = true;

    begin = clock();
    success = tester->send_data();
    end = clock();
    // only show copy time if copy is needed
    if (tester->is_remote()) {
      showTiming("Time to copy data to device", begin, end, opts.verbose());
    }

    if (success) {
      begin = clock();
      success = tester->execute();
      end = clock();
      showTiming("Time to execute plan", begin, end, opts.verbose());

      if (success) {
        begin = clock();
        success = tester->recv_data();
        end = clock();
        // only show copy time if copy is needed
        if (tester->is_remote()) {
          showTiming("Time to copy data to host", begin, end, opts.verbose());
        }

        if (success) {
          begin = clock();
          tester->display(opts.maxprint());
          end = clock();
          showTiming("Time to display results", begin, end, opts.verbose());
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
  tester->destroy_plan();
  end = clock();
  showTiming("Time to teardown/cleanup plan", begin, end, opts.verbose());

  finish = clock(); // algo end
  showTiming("Total time to complete algorithm", start, finish, opts.verbose());
  return retcode;
}

int main(int argc, char *argv[]) {
  clock_t start, finish;

  OptionParser opts{FFT_NAME " " FFT_VERSION};
  if (opts.parse(argc, argv)) {
    std::cout << opts;
  } else {
    return opts.exit_code();
  }

  start = clock(); // algo start

  Tester* tester = TesterFactory::create(opts.type(),
                                         opts.parallelization(),
                                         0,
                                         opts.verbose());
  if (tester) {
    start = clock(); // iterations start
    for (unsigned long i=0; i<opts.iterations(); ++i) {
      if (opts.verbose())
        std::cout << " *** Starting iteration: " << i << " ***" << std::endl;
      else
        std::cout << " *** Current iteration: " << i << " ***\r" << std::flush;
      RetCodes retcode = run_fft(tester, opts);
      if (retcode != SUCCESS) {
        std::cerr << " *** Iteration " << i << " failed with return code: "
                  << retcode << " ***" << std::endl;
        return retcode;
      }
      if (opts.verbose())
        std::cout << " *** Finished iteration: " << i << " ***" << std::endl;
    }

    finish = clock(); // iterations end
    showTiming("Total time to complete all iterations", start, finish);

    // cleanup the tester
    delete tester;

    return SUCCESS;
  } else {
    std::cerr << "Unsupported fft algorithm type: " << opts.type() << std::endl;
    return BAD_ALGO;
  }
}
