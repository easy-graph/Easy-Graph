#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#include "common.h"

#include "utils.cuh"

static __device__ __forceinline__ float atomicMinFloat (
    _OUT_ float * addr,
    _IN_ float value
)
{
    float old;
    old = (value >= 0) ? __int_as_float(atomicMin((int *)addr, __float_as_int(value))) :
        __uint_as_float(atomicMax((unsigned int *)addr, __float_as_uint(value)));

    return old;
}



static __global__ void d_dijkstra (
    _IN_ int32_t* d_V,
    _IN_ int32_t* d_E,
    _IN_ float* d_W,
    _IN_ int32_t* d_sources,
    _IN_ float* d_dist_2D,
    _IN_ int32_t* d_U_2D,
    _IN_ int32_t* d_F_2D,
    _IN_ int32_t len_V,
    _IN_ int32_t len_E,
    _IN_ int32_t len_sources,
    _OUT_ float* d_CC
)
{
    for (int s_idx = blockIdx.x; s_idx < len_sources; s_idx += gridDim.x) {
        int s = d_sources[s_idx];

        int* d_U = d_U_2D + blockIdx.x * len_V;
        int* d_F = d_F_2D + blockIdx.x * len_V;
        float* d_dist = d_dist_2D + blockIdx.x * len_V;

        __shared__ int32_t len_F;
        __shared__ float delta;
        __shared__ float dist_accum;
        __shared__ int32_t reachable_cnt;

        for (int i = threadIdx.x; i < len_V; i += blockDim.x) {
            d_U[i] = 1;
            d_dist[i] = EG_FLOAT_INF;
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            d_dist[s] = 0.0;
            d_F[0] = s;
            len_F = 1;
            delta = 0.0;
            dist_accum = 0.0;
            reachable_cnt = 0;
        }
        __syncthreads();

        while (delta < EG_FLOAT_INF) {
            for (int f_idx = threadIdx.x; f_idx < len_F; f_idx += blockDim.x) {
                int f = d_F[f_idx];
                int edge_start = d_V[f];
                int edge_end = f + 1 < len_V ? d_V[f + 1] : len_E;
                for (int i = edge_start; i < edge_end; ++i) {
                    atomicMinFloat(d_dist + d_E[i], d_dist[f] + d_W[i]);
                }
            }
            __syncthreads();

            if (threadIdx.x == 0) {
                delta = EG_FLOAT_INF;
            }
            __syncthreads();

            for (int i = threadIdx.x; i < len_V; i += blockDim.x) {
                float curr_min = EG_FLOAT_INF;
                if (d_U[i]) {
                    int edge_start = d_V[i];
                    int edge_end = i + 1 < len_V ? d_V[i + 1] : len_E;
                    for (int e = edge_start; e < edge_end; ++e) {
                        curr_min = min(curr_min, d_dist[i] + d_W[e]);
                    }
                }
                atomicMinFloat(&delta, curr_min);
            }
            __syncthreads();

            if (threadIdx.x == 0) {
				len_F = 0;
			}
			__syncthreads();

            for (int i = threadIdx.x; i < len_V; i += blockDim.x) {
                if (d_U[i] && d_dist[i] <= delta && d_dist[i] < EG_FLOAT_INF) {
                    d_U[i] = 0;
                    int f_idx = atomicAdd(&len_F, 1);
                    d_F[f_idx] = i;

                    atomicAdd(&reachable_cnt, 1);
                    atomicAdd(&dist_accum, d_dist[i]);
                }
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            d_CC[s_idx] = IS_EQUAL(dist_accum, 0.0f) ? 0.0 :
                                (double)(reachable_cnt - 1) * 
                                (double)(reachable_cnt - 1) /
                                ((len_V - 1) * dist_accum);
        }
        __syncthreads();
    }
}



// we here use CSR to represent a graph
int cuda_closeness_centrality (
    _IN_ int32_t* V,
    _IN_ int32_t* E,
    _IN_ float* W,
    _IN_ int32_t* sources,
    _IN_ int32_t len_V,
    _IN_ int32_t len_E,
    _IN_ int32_t len_sources,
    _OUT_ float* CC
)
{
    int cuda_ret = cudaSuccess;
    int EG_ret = EG_GPU_SUCC;

    int32_t block_size = 256;
    int32_t grid_size = len_sources / block_size + (len_sources % block_size != 0);

    int32_t *d_V = NULL, *d_E = NULL, *d_sources= NULL;
    int32_t *d_U_2D = NULL, *d_F_2D = NULL;
    float *d_W = NULL, *d_dist_2D = NULL, *d_CC = NULL;

    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_V, sizeof(int32_t) * len_V));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_E, sizeof(int32_t) * len_E));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_sources, sizeof(int32_t) * len_sources));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_U_2D, sizeof(int32_t) * grid_size * len_V));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_F_2D, sizeof(int32_t) * grid_size * len_V));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_W, sizeof(float) * len_E));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_dist_2D, sizeof(float) * grid_size * len_V));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_CC, sizeof(float) * len_V));

    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_V, V, sizeof(int32_t) * len_V, cudaMemcpyHostToDevice));
    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_E, E, sizeof(int32_t) * len_E, cudaMemcpyHostToDevice));
    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_sources, sources, sizeof(int32_t) * len_sources, cudaMemcpyHostToDevice));
    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_W, W, sizeof(float) * len_E, cudaMemcpyHostToDevice));

    d_dijkstra<<<grid_size, block_size>>>(d_V, d_E, d_W, d_sources, d_dist_2D,
                                            d_U_2D, d_F_2D, len_V, len_E, 
                                            len_sources, d_CC);

    EXIT_IF_CUDA_FAILED(cudaMemcpy(CC, d_CC, sizeof(float) * len_V, cudaMemcpyDeviceToHost));

exit:
    cudaFree(d_V);
    cudaFree(d_E);
    cudaFree(d_sources);
    cudaFree(d_U_2D);
    cudaFree(d_F_2D);
    cudaFree(d_W);
    cudaFree(d_dist_2D);
    cudaFree(d_CC);

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