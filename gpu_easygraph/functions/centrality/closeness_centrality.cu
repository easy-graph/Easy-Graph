#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#include "common.h"
#include "utils.h"

__device__ __forceinline__ float atomicMinFloat (
    _OUT_ float * addr,
    _IN_ float value
)
{
    float old;
    old = (value >= 0) ? __int_as_float(atomicMin((int *)addr, __float_as_int(value))) :
        __uint_as_float(atomicMax((unsigned int *)addr, __float_as_uint(value)));

    return old;
}

static __global__ void init_cuda_arr (
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

static __global__ void init_cuda_arr (
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

static __global__ void set_cuda_arr (
    _OUT_ float* d_arr, 
    _IN_ int len, 
    _IN_ float val
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < len) {
        d_arr[tid] = val;
    }
}

static __global__ void relax (
    _IN_ int32_t* d_V,
    _IN_ int32_t* d_E,
    _IN_ float* d_W,
    _IN_ int32_t len_V,
    _IN_ int32_t len_E,
    _IN_ int32_t* d_U,
    _IN_ int32_t* d_F,
    _OUT_ float* d_delta
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < len_V && d_F[tid]) {
        int edge_start = d_V[tid];
        int edge_end = tid + 1 < len_V ? d_V[tid + 1] : len_E;
        for (int i = edge_start; i < edge_end; ++i) {
            if (d_U[d_E[i]] && d_delta[d_E[i]] > d_delta[tid] + d_W[i]) {
                atomicMinFloat(d_delta + d_E[i], d_delta[tid] + d_W[i]);
            }
        }
    }
}

static __global__ void min_delta_demarcation (
    _IN_ int32_t* d_V, 
    _IN_ int32_t* d_E, 
    _IN_ float* d_W,
    _IN_ int32_t len_V, 
    _IN_ int32_t len_E,
    _IN_ int32_t* d_U, 
    _IN_ int32_t* d_F, 
    _IN_ float* d_delta,
    _OUT_ float *min_delta
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= len_V) {
        return;
    }

    if (d_U[tid]) {
        float curr_min = EG_FLOAT_INF;

        int edge_start = d_V[tid];
        int edge_end = tid + 1 < len_V ? d_V[tid + 1] : len_E;
        for (int i = edge_start; i < edge_end; ++i) {
            curr_min = min(curr_min, d_delta[tid] + d_W[i]);
        }

        if (curr_min < *min_delta) {
            atomicMinFloat(min_delta, curr_min);
        }
    }
}

static __global__ void update (
    _IN_ int32_t len_V, 
    _OUT_ int32_t* d_U, 
    _OUT_ int32_t* d_F, 
    _IN_ float* d_delta,
    _IN_ float* d_min_delta
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= len_V) {
        return;
    }

    if (d_U[tid] && d_delta[tid] <= *d_min_delta) {
        d_U[tid] = 0;
        d_F[tid] = 1;
    }
}

static int dijkstra (
    _IN_ int32_t* d_V, 
    _IN_ int32_t* d_E, 
    _IN_ float* d_W, 
    _IN_ int32_t len_V, 
    _IN_ int32_t len_E, 
    _IN_ int32_t source, 
    _OUT_ float* d_delta
)
{
    int cuda_ret = cudaSuccess;

    // define vars
    // d_U means unsettled, d_F means frontier
    int32_t *d_U, *d_F;
    int32_t block_size = 512;
    int32_t grid_size = (len_V + block_size) / block_size;
    float *d_min_delta;
    float h_min_delta;
    const float float_max_inst = EG_FLOAT_INF;

    // initialize vars
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_U, sizeof(int32_t) * len_V));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_F, sizeof(int32_t) * len_V));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_min_delta, sizeof(float)));
    init_cuda_arr<<<grid_size, block_size>>>(d_U, 1, len_V, source, 0);
    init_cuda_arr<<<grid_size, block_size>>>(d_F, 0, len_V, source, 1);
    init_cuda_arr<<<grid_size, block_size>>>(d_delta, EG_FLOAT_INF, len_V, source, 0.0f);

    // relax, get min, and update
    while (1) {
        grid_size = (len_V + block_size) / block_size;
        relax<<<grid_size, block_size>>>(d_V, d_E, d_W, len_V, len_E, d_U, d_F, d_delta);

        EXIT_IF_CUDA_FAILED(cudaMemcpy(d_min_delta, &float_max_inst, sizeof(float), cudaMemcpyHostToDevice));
        min_delta_demarcation<<<grid_size, block_size>>>(d_V, d_E, d_W, len_V,
                                                    len_E, d_U, d_F, d_delta, d_min_delta);

        update<<<grid_size, block_size>>>(len_V, d_U, d_F, d_delta, d_min_delta);

        EXIT_IF_CUDA_FAILED(cudaMemcpy(&h_min_delta, d_min_delta, sizeof(float), cudaMemcpyDeviceToHost));

        if (IS_EQUAL(h_min_delta, EG_FLOAT_INF)) {
            break;
        }
    }
exit:
    cudaFree(d_U);
    cudaFree(d_F);
    cudaFree(d_min_delta);
    return cuda_ret;
}

// All-Pairs Shortest Path
static int APSP (
    _IN_ int32_t* V, 
    _IN_ int32_t* E, 
    _IN_ float* W, 
    _IN_ int32_t* sources,
    _IN_ int32_t len_V, 
    _IN_ int32_t len_E,
    _OUT_ float_t* d_apsp
)
{
    int cuda_ret = cudaSuccess;

    int32_t *d_V = NULL, *d_E = NULL;
    float *d_W = NULL;
    
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_V, sizeof(int32_t) * len_V));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_E, sizeof(int32_t) * len_E));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_W, sizeof(float) * len_E));

    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_V, V, sizeof(int32_t) * len_V, cudaMemcpyHostToDevice));
    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_E, E, sizeof(int32_t) * len_E, cudaMemcpyHostToDevice));
    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_W, W, sizeof(float) * len_E, cudaMemcpyHostToDevice));
    
    for (int s = 0; s < len_V; ++s) {
        if (sources[s] == EG_GPU_NODE_ACTIVE) {
            EXIT_IF_CUDA_FAILED(dijkstra(d_V, d_E, d_W, len_V, len_E, s, d_apsp + s * len_V));
        }
    }

exit:
    cudaFree(d_V);
    cudaFree(d_E);
    cudaFree(d_W);
    return cuda_ret;
}

static __global__ void calc_CC (
    _IN_ float* d_apsp, 
    _IN_ int32_t* d_sources,
    _IN_ int32_t len_V, 
    _OUT_ float* d_CC
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= len_V || d_sources[tid] == EG_GPU_NODE_INACTIVE) {
        return;
    }

    int32_t conn_cnt = 0;
    float accum = 0.0f;
    for (int i = 0; i < len_V; ++i) {
        float dist = d_apsp[tid * len_V + i];
        if (!IS_EQUAL(dist, EG_FLOAT_INF)) {
            accum += dist;
            ++conn_cnt;
        }
    }
    d_CC[tid] = IS_EQUAL(accum, 0.0f) ? 0.0f :
            (conn_cnt - 1) * (conn_cnt - 1) / ((len_V - 1) * accum);
}

// we here use CSR to represent a graph
int cuda_closeness_centrality (
    _IN_ int32_t* V, 
    _IN_ int32_t* E, 
    _IN_ float* W, 
    _IN_ int32_t* sources,
    _IN_ int32_t len_V, 
    _IN_ int32_t len_E,  
    _OUT_ float* CC
)
{
    int cuda_ret = cudaSuccess;
    int EG_ret = EG_GPU_SUCC;
    int32_t block_size = 512;
    int32_t grid_size = (len_V + block_size) / block_size;
    float* d_apsp = NULL; // a 2d arr arranged in 1d
    float* d_CC = NULL;
    int* d_sources = NULL;

    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_apsp, sizeof(float*) * len_V * len_V));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_CC, sizeof(float*) * len_V));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_sources, sizeof(int*) * len_V));

    set_cuda_arr<<<grid_size, block_size>>>(d_apsp, len_V * len_V, EG_FLOAT_INF);
    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_sources, sources, sizeof(int32_t) * len_V, cudaMemcpyHostToDevice));

    EXIT_IF_CUDA_FAILED(APSP(V, E, W, sources, len_V, len_E, d_apsp));

    calc_CC<<<grid_size, block_size>>>(d_apsp, d_sources, len_V, d_CC);

    EXIT_IF_CUDA_FAILED(cudaMemcpy(CC, d_CC, sizeof(float) * len_V, cudaMemcpyDeviceToHost));

exit:
    cudaFree(d_apsp);
    cudaFree(d_CC);
    cudaFree(d_sources);

    if (cuda_ret != cudaSuccess) {
        switch (cuda_ret) {
            case cudaErrorMemoryAllocation:
                EG_ret = EG_GPU_FAILED_TO_ALLOCATE_DEVICE_MEM;
                break;
            default:
                EG_ret = EG_GPU_DEVICE_ERR;
                break;
        }
    }

    return EG_ret;
}