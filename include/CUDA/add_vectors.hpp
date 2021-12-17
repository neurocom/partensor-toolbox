#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define CUDA_VEC_BLOCK_SIZE 1024

extern "C" void cuda_vecAdd(double *A, double *B, double *C, int length);
