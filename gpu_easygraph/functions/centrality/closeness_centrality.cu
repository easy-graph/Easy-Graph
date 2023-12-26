#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#include "common.h"

static __device__ double atomicAddDouble (
    _OUT_ double* address, 
    _IN_ double val
)
{
	unsigned long long int* address_as_ull =
		(unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
			__double_as_longlong(val +
			__longlong_as_double(assumed)));
	} while (assumed != old);
	return __longlong_as_double(old);
}



static __device__ double atomicMinDouble (
    _OUT_ double *address, 
    _IN_ double val
)
{
	unsigned long long ret = __double_as_longlong(*address);
	while (val < __longlong_as_double(ret))
	{
		unsigned long long old = ret;
		if ((ret = atomicCAS((unsigned long long *)address, old, __double_as_longlong(val))) == old)
			break;
	}
	return __longlong_as_double(ret);
}



static __global__ void d_calc_min_edge (
    _IN_ int* d_V,
    _IN_ int* d_E,
    _IN_ double* d_W,
    _IN_ int len_V,
    _IN_ int len_E,
    _OUT_ double* d_min_edge
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < len_V) {
		double curr_min = EG_DOUBLE_INF;
        int edge_start = d_V[tid];
        int edge_end = tid + 1 < len_V ? d_V[tid + 1] : len_E;
		for(int i = edge_start; i < edge_end; ++i) {
            curr_min = min(curr_min, d_W[i]);
		}
		d_min_edge[tid] = curr_min;
	}
}

static __global__ void d_dijkstra_cc (
    _IN_ int* d_V,
    _IN_ int* d_E,
    _IN_ double* d_W,
    _IN_ double* d_min_edge,
    _IN_ int* d_sources,
    _BUFFER_ double* d_dist_2D,
    _BUFFER_ int* d_U_2D,
    _BUFFER_ int* d_F_2D,
    _IN_ int len_V,
    _IN_ int len_E,
    _IN_ int len_sources,
    _IN_ int warp_size,
    _OUT_ double* d_CC
)
{
    for (int s_idx = blockIdx.x; s_idx < len_sources; s_idx += gridDim.x) {
        int s = d_sources[s_idx];

        int* d_U = d_U_2D + blockIdx.x * len_V;
        int* d_F = d_F_2D + blockIdx.x * len_V;
        double* d_dist = d_dist_2D + blockIdx.x * len_V;

        __shared__ int len_F;
        __shared__ double delta;
        __shared__ double dist_accum;
        __shared__ int reachable_cnt;

        for (int i = threadIdx.x; i < len_V; i += blockDim.x) {
            d_U[i] = 1;
            d_dist[i] = EG_DOUBLE_INF;
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

        while (delta < EG_DOUBLE_INF) {
            for (int j = threadIdx.x; j < len_F * warp_size; j += blockDim.x) {
                int f = d_F[j / warp_size];
                int edge_start = d_V[f];
                int edge_end = f + 1 < len_V ? d_V[f + 1] : len_E;
                double dist = d_dist[f];
                for (int e = j % warp_size; e < edge_end - edge_start; e += warp_size) {
                    int adj = d_E[e + edge_start];
                    double relax_w = dist + d_W[e + edge_start];
                    atomicMinDouble(d_dist + adj, relax_w);
                }
                __threadfence_block();
            }
            __syncthreads();

            if (threadIdx.x == 0) {
                delta = EG_DOUBLE_INF;
            }
            __syncthreads();

            for (int i = threadIdx.x; i < len_V; i += blockDim.x) {
                double dist_i = d_dist[i];
                if (d_U[i] == 1 && dist_i < EG_DOUBLE_INF) {
                    atomicMinDouble(&delta, dist_i + d_min_edge[i]);
                }
            }
            __syncthreads();

            if (threadIdx.x == 0) {
				len_F = 0;
			}
			__syncthreads();

            for (int i = threadIdx.x; i < len_V; i += blockDim.x) {
                double dist_i = d_dist[i];
                if (d_U[i] && dist_i <= delta && dist_i < EG_DOUBLE_INF) {
                    d_U[i] = 0;
                    int f_idx = atomicAdd(&len_F, 1);
                    d_F[f_idx] = i;

                    atomicAdd(&reachable_cnt, 1);
                    atomicAddDouble(&dist_accum, d_dist[i]);
                }
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            d_CC[s_idx] = dist_accum == 0.0 ? 0.0 :
                                (double)(reachable_cnt - 1) * 
                                (double)(reachable_cnt - 1) /
                                ((len_V - 1) * dist_accum);
        }
        __syncthreads();
    }
}



// we here use CSR to represent a graph
int cuda_closeness_centrality (
    _IN_ int* V,
    _IN_ int* E,
    _IN_ double* W,
    _IN_ int* sources,
    _IN_ int len_V,
    _IN_ int len_E,
    _IN_ int len_sources,
    _IN_ int warp_size,
    _OUT_ double* CC
)
{
    int cuda_ret = cudaSuccess;
    int EG_ret = EG_GPU_SUCC;

    int block_size = 256;
    size_t grid_size = len_V / block_size + (len_V % block_size != 0);
    size_t mem_free = 0, mem_total = 0;

    int *d_V = NULL, *d_E = NULL, *d_sources= NULL;
    int *d_U_2D = NULL, *d_F_2D = NULL;
    double *d_W = NULL, *d_min_edge = NULL, *d_dist_2D = NULL, *d_CC = NULL;

    EXIT_IF_CUDA_FAILED(cudaMemGetInfo(&mem_free, &mem_total));
    while (true) {
        size_t mem_needed = sizeof(int) * len_V // d_V
                        + sizeof(int) * len_E // d_E
                        + sizeof(int) * len_sources // d_sources
                        + sizeof(int) * grid_size * len_V // d_U_2D
                        + sizeof(int) * grid_size * len_V // d_F_2D
                        + sizeof(double) * len_E // d_W
                        + sizeof(double) * len_V // d_min_edge
                        + sizeof(double) * grid_size * len_V // d_dist_2D
                        + sizeof(double) * len_V // d_BC
                        ;
        if (mem_needed < mem_free / 2) {
            break;
        } else {
            grid_size /= 2;
        }
    }

    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_V, sizeof(int) * len_V));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_E, sizeof(int) * len_E));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_sources, sizeof(int) * len_sources));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_U_2D, sizeof(int) * grid_size * len_V));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_F_2D, sizeof(int) * grid_size * len_V));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_W, sizeof(double) * len_E));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_min_edge, sizeof(double) * len_V));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_dist_2D, sizeof(double) * grid_size * len_V));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_CC, sizeof(double) * len_V));

    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_V, V, sizeof(int) * len_V, cudaMemcpyHostToDevice));
    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_E, E, sizeof(int) * len_E, cudaMemcpyHostToDevice));
    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_sources, sources, sizeof(int) * len_sources, cudaMemcpyHostToDevice));
    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_W, W, sizeof(double) * len_E, cudaMemcpyHostToDevice));

    d_calc_min_edge<<<grid_size, block_size>>>(d_V, d_E, d_W, len_V, len_E, d_min_edge);

    d_dijkstra_cc<<<grid_size, block_size>>>(d_V, d_E, d_W, d_min_edge, d_sources, d_dist_2D,
                                    d_U_2D, d_F_2D, len_V, len_E, len_sources, warp_size, d_CC);

    EXIT_IF_CUDA_FAILED(cudaMemcpy(CC, d_CC, sizeof(double) * len_V, cudaMemcpyDeviceToHost));

exit:
    cudaFree(d_V);
    cudaFree(d_E);
    cudaFree(d_sources);
    cudaFree(d_U_2D);
    cudaFree(d_F_2D);
    cudaFree(d_W);
    cudaFree(d_min_edge);
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