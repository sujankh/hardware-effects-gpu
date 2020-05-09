#include <cstdio>
#include <device_launch_parameters.h>

#include "../common.h"

//#define REPETITIONS 1000
#define REPETITIONS 1
#define MEMORY_SIZE 4096

using Type = uint32_t;

__global__ void kernel(int offset, float* out)
{
    int threadId = threadIdx.x;

    float sum = 0;

    if (threadId < offset)
      {
    	for(int i = 0; i < 10000; ++i)
    	  {
    	    sum += cosf(floorf(sqrtf(threadId + i)));
    	  }
      }
    else
      {
    	for(int i = 0; i < 10000; ++i)
    	  {
    	    sum += cosf(floorf(sqrtf(threadId + i)));
    	  }
      }

    __syncthreads();
    out[threadId] = sum;
}

__global__ void kernel_predication(int offset, float* out)
{
    int threadId = threadIdx.x;

    float sum = 0;
    //sum = threadId;
    if (threadId < offset)
      {
	sum = (float)clock64();
      }
    else
      {
	sum += (float)clock64();
      }

    __syncthreads();
    out[threadId] = sum;
}


static void benchmark(int offset)
{
    float time = 0;
    float *out;  // The kernel will output to this buffer
    cudaMallocManaged(&out, sizeof(float) * 32);

    for (int i = 0; i < REPETITIONS; i++)
    {
        CudaTimer timer;
        kernel<<<1, 32>>>(offset, out);  // launch exactly one warp
	//kernel_predication<<<1, 32>>>(offset, out);  // launch exactly one warp
        CHECK_CUDA_CALL(cudaPeekAtLastError());
        timer.stop_wait();
        time += timer.get_time();
    }

    std::cerr << time / REPETITIONS << std::endl;
}

void run(int num_diverge_threads)
{
    auto prop = initGPU();
    std::cout << "Warp size: " << prop.warpSize << std::endl;

    // // query shared memory bank size
    // cudaSharedMemConfig sharedMemConfig;
    // CHECK_CUDA_CALL(cudaDeviceGetSharedMemConfig(&sharedMemConfig));
    // std::cout << "Bank size: " << (sharedMemConfig == cudaSharedMemBankSizeEightByte ? 8 : 4) << std::endl;

    // // set it to four, just in case
    // CHECK_CUDA_CALL(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte));

    benchmark(num_diverge_threads);
}
