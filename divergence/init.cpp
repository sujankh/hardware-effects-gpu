#include <iostream>
#include "divergence-benchmark.h"

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::cout << "Usage: bank-conflicts <offset>" << std::endl;
        return 1;
    }

    std::string benchmarkType(argv[1]);
    std::cout << benchmarkType << std::endl;

    DivergenceBenchmark bench;
    bench.setup();

    if (benchmarkType == "predication")
      {
	//nv-nsight-cu-cli --metrics "smsp__thread_inst_executed.sum" ./divergence/divergence predicate <true_block_count>
	const int true_block_count = std::stoi(argv[2]);
	bench.run_predication_benchmark(true_block_count);
      }
    else if (benchmarkType == "divergence")
      {
	//nv-nsight-cu-cli --metrics "smsp__thread_inst_executed.sum" ./divergence/divergence predicate <true_block_count>
	const int true_block_count = std::stoi(argv[2]);
	bench.run_divergence_benchmark(true_block_count);
      }


    bench.print_stats();
    bench.tear_down();

    return 0;
}
