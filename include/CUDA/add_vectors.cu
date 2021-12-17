// #include "cuda.h"
// #include "cuda_runtime.h"
// #include "device_launch_parameters.h"
#include "add_vectors.hpp"

__global__ void vecAdd(double *a, double *b, double *c, int n)
{
    // Get our global thread ID
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    // Make sure we do not go out of bounds
    if (id < n)
        c[id] = a[id] + b[id];
}

extern "C" void cuda_vecAdd(double *A, double *B, double *C, int length)
{
    int blockSize, gridSize;
    blockSize = CUDA_VEC_BLOCK_SIZE;

    // Number of thread blocks in grid
    gridSize = (int)ceil((float)length / blockSize);

    vecAdd<<<gridSize, blockSize>>>(A, B, C, length);
}   