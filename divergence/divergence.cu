#include <algorithm>
#include <cstdio>
#include <device_launch_parameters.h>

#include "../common.h"
#include "divergence-benchmark.h"

#define REPETITIONS 1

__global__ void branch_predication(int true_block_count, float *out, long long int *timer) {
  int threadId = blockDim.x * blockIdx.x + threadIdx.x;
  float sum = 0;

  // Warmup
  #pragma unroll 1
  for (int i = 0; i < 10; ++i) {
    if (threadId < true_block_count)
      {
	sum += i;
      }
    else
      {
	sum += i + 11;                  // @P0 FADD   ...
	sum -= i * sum;                 // @P0 FFMA   ...
	sum -= sum * sum;               // @P0 FFMA   ...
      }
  }

  long long stop;
  long long start = clock64();

  // Disable full unroll of loop to make it easier to read the SASS output
  #pragma unroll 1
  for (int i = 0; i < 100; ++i) {

    // The following if-else gets converted to predicated instructions so all
    // the instructions get scheduled for execution

    // From: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#branch-predication
    // This is a simple if-else so with less than 4 instr, so the compiler tends to produce predicated
    // instructions instead of doing an actual branch


    if (threadId < true_block_count)
      {
	// Total instr = 3 (From nsight compute)
	sum += i;                        // @!P0 FADD .... and 2 other instr
      }
    else
      {
	// Total instr = 5 (From nsight compute)
	sum += i + 11;                  // @P0 FADD   ... and 2 other instr
	sum -= i * sum;                 // @P0 FFMA   ...
	sum -= sum * sum;               // @P0 FFMA   ...
      }
  }

  stop = clock64();
  __syncthreads();
  timer[threadId] = stop - start;
  // Write the output to make sure it is not optimized out
  out[threadId] = sum;
}


__global__ void divergence(int true_block_count, float *out, long long int *timer) {
  int threadId = blockDim.x * blockIdx.x + threadIdx.x;
  float sum = 0;

  // Warmup

  #pragma unroll 1
  for (int i = 0; i < 10; ++i) {

    if (threadId < true_block_count)
      {
	sum += __sinf(i);
	sum += __cosf(i+5);
      }
    else
      {
	sum += i + sqrtf(i - 22);
      }
  }

  // #pragma unroll 1
  // for (int i = 0; i < 10; ++i) {
  //   if (threadId < true_block_count)
  //     {
  // 	sum += i;
  //     }
  //   else
  //     {
  // 	sum += i + 11;                  // @P0 FADD   ...
  // 	sum -= i * sum;                 // @P0 FFMA   ...
  // 	sum -= sum * sum;               // @P0 FFMA   ...
  //     }
  // }

  long long stop;
  long long start = clock64();

  // Disable full unroll of loop to make it easier to read the SASS output
  #pragma unroll 1
  for (int i = 0; i < 100; ++i) {

    if (threadId < true_block_count)
      {
	sum += __sinf(i);
	sum += __cosf(i+5);
      }
    else
      {
	sum += i + sqrtf(i - 22);
      }
  }

  stop = clock64();
  __syncthreads();
  timer[threadId] = stop - start;
  // Write the output to make sure it is not optimized out
  out[threadId] = sum;
}


// __global__ void kernel(int *loop_limit, float *out, long long int *timer) {
//   int threadId = blockDim.x * blockIdx.x + threadIdx.x;
//   int limit = loop_limit[threadId];
//   float sum = 0;

//   // Warmup
// // #pragma unroll 1
// //   for (int i = 0; i < limit; ++i) {
// //     sum += __sinf(i);
// //   }

//   long long start = clock64();
//   long long stop;

//   //__sinf = ~52 cycles on a GTX 1660
//   for (int k = 0; k < REPETITIONS; ++k) {
//     // unroll 1 to make sure the loop is not completely unrolled
//     // else there won't be any branches
// #pragma unroll 1
//     for (int i = 0; i < 100; ++i) {

//       // if (threadId < 0)
//       // 	sum += i;
//       // else
//       // 	{
//       // 	sum += i + 11;
//       // 	sum += __sinf(i);
//       // 	}


//       if (threadId < 10) {
//       	sum += __sinf(i);
//       	sum += __cosf(i+5);
//       }
//       else
//       	{
//       	  sum += i + sqrtf(i - 22);
//       	}
//     }
//   }

//   stop = clock64();
//   __syncthreads();
//   timer[threadId] = stop - start;
//   // Write the output to make sure it is not optimized out
//   out[threadId] = sum;
// }


// class DivergenceBenchmark
// {
// public:

//   void setup()
//   {
//     cudaMallocManaged(&out, sizeof(float) * NUM_THREADS);
//     cudaMallocManaged(&timer, sizeof(long long int) * NUM_THREADS);
//     CHECK_CUDA_CALL(cudaMemset(timer, 0, sizeof(long long int) * NUM_THREADS));
//   }

//   // nv-nsight-cu-cli --metrics "smsp__thread_inst_executed.sum" ./divergence/divergence 32
//   void run_predicate_benchmark(int true_block_count)
//   {
//         predication<<<1, 32>>>(true_block_count, out, timer); // launch exactly one warp
// 	cudaDeviceSynchronize();
// 	CHECK_CUDA_CALL(cudaPeekAtLastError());
//   }

//   void print_stats()
//   {
//     std::cout << "Total cycles:" << timer[0] << std::endl;
//   }

//   void tear_down()
//   {
//     cudaFree(out);
//     cudaFree(timer);
//   }

// private:
//   float *out;           // The kernel will output to this buffer
//   long long int *timer; // Time for each thread to run the loop
//   constexpr int NUM_THREADS = 32;
// };

  void DivergenceBenchmark::setup()
  {
    cudaMallocManaged(&out, sizeof(float) * NUM_THREADS);
    cudaMallocManaged(&timer, sizeof(long long int) * NUM_THREADS);
    CHECK_CUDA_CALL(cudaMemset(timer, 0, sizeof(long long int) * NUM_THREADS));
  }

  // nv-nsight-cu-cli --metrics "smsp__thread_inst_executed.sum" ./divergence/divergence 32
  void DivergenceBenchmark::run_predication_benchmark(int true_block_count)
  {
    std::cout << "Total threads that will run the true block(3 instr): " << true_block_count << std::endl;
    std::cout << "Total threads that will run the false block(5 instr): " << (NUM_THREADS - true_block_count) << std::endl;

    branch_predication<<<1, 32>>>(true_block_count, out, timer); // launch exactly one warp
    cudaDeviceSynchronize();
    CHECK_CUDA_CALL(cudaPeekAtLastError());
  }

  void DivergenceBenchmark::run_divergence_benchmark(int true_block_count)
  {
    std::cout << "Total threads that will run the true block(3 instr): " << true_block_count << std::endl;
    std::cout << "Total threads that will run the false block(5 instr): " << (NUM_THREADS - true_block_count) << std::endl;

    divergence<<<1, 32>>>(true_block_count, out, timer); // launch exactly one warp
    cudaDeviceSynchronize();
    CHECK_CUDA_CALL(cudaPeekAtLastError());
  }

  void DivergenceBenchmark::print_stats()
  {
    std::cout << "Total cycles:" << timer[0] << std::endl;
  }

  void DivergenceBenchmark::tear_down()
  {
    cudaFree(out);
    cudaFree(timer);
  }


// static void benchmark(int num_divergent_threads) {
//   float *out;           // The kernel will output to this buffer
//   long long int *timer; // Time for each thread to run the loop
//                         // std::vector<long long int>
//   int *loop_limit;      // Max num of times the loop can run for each thread


//   cudaMallocManaged(&out, sizeof(float) * NUM_THREADS);
//   cudaMallocManaged(&loop_limit, sizeof(int) * NUM_THREADS);
//   cudaMallocManaged(&timer, sizeof(long long int) * NUM_THREADS);
//   CHECK_CUDA_CALL(cudaMemset(timer, 0, sizeof(long long int) * NUM_THREADS));

//   // Set loop limit for each thread = 32 if no threads diverge
//   std::fill(loop_limit, loop_limit + NUM_THREADS, 32);

//   // Update loop_limit such that `num_divergent_threads` elements in it have
//   // different value than 32
//   for (int i = 0, k = num_divergent_threads; k > 0; --k, ++i) {
//     loop_limit[i] = loop_limit[i] - k;
//   }

//   for (int i = 0; i < REPETITIONS; i++) {
//     predication<<<1, 32>>>(loop_limit, out, timer); // launch exactly one warp
//     cudaDeviceSynchronize();
//     CHECK_CUDA_CALL(cudaPeekAtLastError());
//   }

//   for (int i = 0; i < NUM_THREADS; ++i) {
//     std::cout << loop_limit[i] << "  ";
//   }

//   std::cout << "     :" << timer[0] << std::endl;

//   cudaFree(out);
//   cudaFree(timer);
// }

// void run(int num_diverge_threads) {
//   auto prop = initGPU();
//   std::cout << "Warp size: " << prop.warpSize << std::endl;
//   std::cout << "Diverging threads: " << num_diverge_threads << std::endl;

//   benchmark(num_diverge_threads);
// }
