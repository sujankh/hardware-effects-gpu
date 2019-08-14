#include <algorithm>
#include <cstdio>
#include <device_launch_parameters.h>

#include "../common.h"

#define REPETITIONS 10

__global__ void kernel(int *loop_limit, float *out, long long int *timer) {
  int threadId = blockDim.x * blockIdx.x + threadIdx.x;
  int limit = loop_limit[threadId];
  float sum = 0;

  // Warmup
#pragma unroll 1
  for (int i = 0; i < limit; ++i) {
    sum += __sinf(i);
  }

  long long start = clock64();
  long long stop;

  //__sinf = ~52 cycles on a GTX 1660
  for (int k = 0; k < REPETITIONS; ++k) {
    // unroll 1 to make sure the loop is not completely unrolled
    // else there won't be any branches
#pragma unroll 1
    for (int i = 0; i < limit; ++i) {
      sum += __sinf(i);
    }
  }

  stop = clock64();
  __syncthreads();
  timer[threadId] = stop - start;
  // Write the output to make sure it is not optimized out
  out[threadId] = sum;
}

static void benchmark(int num_divergent_threads) {
  float *out;           // The kernel will output to this buffer
  long long int *timer; // Time for each thread to run the loop
                        // std::vector<long long int>
  int *loop_limit;      // Max num of times the loop can run for each thread
  constexpr int NUM_THREADS = 32;

  cudaMallocManaged(&out, sizeof(float) * NUM_THREADS);
  cudaMallocManaged(&loop_limit, sizeof(int) * NUM_THREADS);
  cudaMallocManaged(&timer, sizeof(long long int) * NUM_THREADS);
  CHECK_CUDA_CALL(cudaMemset(timer, 0, sizeof(long long int) * NUM_THREADS));

  // Set loop limit for each thread = 32 if no threads diverge
  std::fill(loop_limit, loop_limit + NUM_THREADS, 32);

  // Update loop_limit such that `num_divergent_threads` elements in it have
  // different value than 32
  for (int i = 0, k = num_divergent_threads; k > 0; --k, ++i) {
    loop_limit[i] = loop_limit[i] - k;
  }

  for (int i = 0; i < REPETITIONS; i++) {
    kernel<<<1, 32>>>(loop_limit, out, timer); // launch exactly one warp
    cudaDeviceSynchronize();
    CHECK_CUDA_CALL(cudaPeekAtLastError());
  }

  for (int i = 0; i < NUM_THREADS; ++i) {
    std::cout << loop_limit[i] << "  ";
  }

  std::cout << "     :" << timer[0] << std::endl;

  cudaFree(out);
  cudaFree(timer);
}

void run(int num_diverge_threads) {
  auto prop = initGPU();
  std::cout << "Warp size: " << prop.warpSize << std::endl;
  std::cout << "Diverging threads: " << num_diverge_threads << std::endl;

  benchmark(num_diverge_threads);
}
