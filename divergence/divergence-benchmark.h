
class DivergenceBenchmark
{
public:
  void setup();
  void run_predication_benchmark(int true_block_count);
  void run_divergence_benchmark(int true_block_count);
  void print_stats();
  void tear_down();
private:
  float *out;           // The kernel will output to this buffer
  long long int *timer; // Time for each thread to run the loop
  const int NUM_THREADS = 32;
};
