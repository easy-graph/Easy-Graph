#include <cuda.h>
#include <cuda_runtime.h>

#include "common.h"

int access_cuda_mem(
    _IN_ int32_t* d_i
);



float access_cuda_mem(
    _IN_ float* d_f
);



int assign_cuda_val(
    _OUT_ float* d_f,
    _IN_ float f
);



int assign_cuda_val(
    _OUT_ int* d_i,
    _IN_ int i
);



__global__ void init_cuda_arr (
    _OUT_ int32_t* d_arr, 
    _IN_ int32_t val, 
    _IN_ int32_t len, 
    _IN_ int32_t source, 
    _IN_ int32_t source_val
);



__global__ void init_cuda_arr (
    _OUT_ float* d_in, 
    _IN_ float val, 
    _IN_ int32_t len, 
    _IN_ int32_t source, 
    _IN_ float source_val
);