#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#include "common.h"

int access_cuda_mem(
    _IN_ int32_t* d_i
)
{
    int i = 0;
    cudaMemcpy(&i, d_i, sizeof(int32_t), cudaMemcpyDeviceToHost);

    return i;
}



float access_cuda_mem(
    _IN_ float* d_f
)
{
    float f = 0;
    cudaMemcpy(&f, d_f, sizeof(float), cudaMemcpyDeviceToHost);

    return f;
}



int assign_cuda_val(
    _OUT_ float* d_f,
    _IN_ float f
)
{
    return cudaMemcpy(d_f, &f, sizeof(float), cudaMemcpyHostToDevice);
}



int assign_cuda_val(
    _OUT_ int* d_i,
    _IN_ int i
)
{
    return cudaMemcpy(d_i, &i, sizeof(int), cudaMemcpyHostToDevice);
}



__global__ void init_cuda_arr (
    _OUT_ int32_t* d_arr, 
    _IN_ int32_t val, 
    _IN_ int32_t len, 
    _IN_ int32_t source, 
    _IN_ int32_t source_val
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < len) {
        d_arr[tid] = val;
    }
    if (tid == source) {
        d_arr[tid] = source_val;
    }
}



__global__ void init_cuda_arr (
    _OUT_ float* d_in, 
    _IN_ float val, 
    _IN_ int32_t len, 
    _IN_ int32_t source, 
    _IN_ float source_val
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < len) {
        d_in[tid] = val;
    }
    if (tid == source) {
        d_in[tid] = source_val;
    }
}