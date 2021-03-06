#include <cstdio>
#include <device_launch_parameters.h>

#include "../common.h"

#define REPETITIONS 1000
#define MEMORY_SIZE 4096

using Type = uint32_t;

__global__ void kernel(Type* memory, int startOffset, int moveOffset)
{
    int threadId = threadIdx.x;

    // repeatedly read and write to global memory
    uint32_t index = threadId * startOffset;
    for (int i = 0; i < 4000; i++)
    {
        memory[index] += index * i;
        index += moveOffset;
        index %= MEMORY_SIZE;
    }
}

static void benchmark(int startOffset, int moveOffset)
{
    CudaMemory<Type> memory(MEMORY_SIZE);

    float time = 0;
    for (int i = 0; i < REPETITIONS; i++)
    {
        CudaTimer timer;
        kernel<<<1, 32>>>(memory.pointer(), startOffset, moveOffset);  // launch exactly one warp
        CHECK_CUDA_CALL(cudaPeekAtLastError());
        timer.stop_wait();
        time += timer.get_time();
    }

    std::cerr << time / REPETITIONS << std::endl;
}

void run(int startOffset, int moveOffset)
{
    initGPU();
    benchmark(startOffset, moveOffset);
}
