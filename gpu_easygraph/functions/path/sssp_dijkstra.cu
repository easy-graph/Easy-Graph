#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#include "common.h"

namespace gpu_easygraph {

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
    int tnum = blockDim.x * gridDim.x;

    for (int u = tid; u < len_V; u += tnum) {
        double curr_min = EG_DOUBLE_INF;
        int edge_start = d_V[u];
        int edge_end = d_V[u + 1];
        for(int v = edge_start; v < edge_end; ++v) {
            curr_min = min(curr_min, d_W[v]);
        }
        d_min_edge[u] = curr_min;
    }
}



static __global__ void d_sssp_dijkstra (
    _IN_ int* d_curr_node,
    _IN_ int* d_V,
    _IN_ int* d_E,
    _IN_ double* d_W,
    _IN_ double* d_min_edge,
    _IN_ int* d_sources,
    _OUT_ double* d_dist_2D,
    _BUFFER_ int* d_U_2D,
    _BUFFER_ int* d_F_2D,
    _IN_ int len_V,
    _IN_ int len_E,
    _IN_ int len_sources,
    _IN_ int target,
    _IN_ int warp_size
)
{
    while (1) {
        __shared__ int curr_node;
        if (threadIdx.x == 0) {
            curr_node = atomicAdd(d_curr_node, 1);
        }
        __syncthreads();

        if (curr_node >= len_sources) {
            break;
        }

        int s = d_sources[curr_node];

        double* d_dist = d_dist_2D + curr_node * len_V;
        int* d_U = d_U_2D + blockIdx.x * len_V;
        int* d_F = d_F_2D + blockIdx.x * len_V;

        __shared__ int len_F;
        __shared__ double delta;
        __shared__ int target_cnt;

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
            target_cnt = 0;
        }
        __syncthreads();

        while (delta < EG_DOUBLE_INF && target_cnt == 0) {
            for (int j = threadIdx.x; j < len_F * warp_size; j += blockDim.x) {
                int f = d_F[j / warp_size];
                int edge_start = d_V[f];
                int edge_end = d_V[f + 1];
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
                    target_cnt += i == target;
                }
            }
            __syncthreads();
        }

        __syncthreads();
    }
}



// we here use CSR to represent a graph
int cuda_sssp_dijkstra(
    _IN_ const int* V,
    _IN_ const int* E,
    _IN_ const double* W,
    _IN_ const int* sources,
    _IN_ int len_V,
    _IN_ int len_E,
    _IN_ int len_sources,
    _IN_ int target,
    _IN_ int warp_size,
    _OUT_ double* res
)
{
    int cuda_ret = cudaSuccess;
    int EG_ret = EG_GPU_SUCC;

    int min_edge_block_size;
    int min_edge_grid_size;
    int dijkstra_block_size;
    int dijkstra_grid_size;

    cudaOccupancyMaxPotentialBlockSize(&min_edge_grid_size, &min_edge_block_size, d_calc_min_edge, 0, 0); 
    cudaOccupancyMaxPotentialBlockSize(&dijkstra_grid_size, &dijkstra_block_size, d_sssp_dijkstra, 0, 0); 

    int *d_curr_node = NULL;
    int *d_V = NULL, *d_E = NULL, *d_sources= NULL;
    int *d_U_2D = NULL, *d_F_2D = NULL;
    double *d_W = NULL, *d_min_edge = NULL, *d_dist_2D = NULL;

    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_curr_node, sizeof(int)));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_V, sizeof(int) * (len_V + 1)));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_E, sizeof(int) * len_E));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_sources, sizeof(int) * len_sources));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_U_2D, sizeof(int) * dijkstra_grid_size * len_V));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_F_2D, sizeof(int) * dijkstra_grid_size * len_V));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_W, sizeof(double) * len_E));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_min_edge, sizeof(double) * len_V));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_dist_2D, sizeof(double) * len_sources * len_V));

    EXIT_IF_CUDA_FAILED(cudaMemset(d_curr_node, 0, sizeof(int)));
    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_V, V, sizeof(int) * (len_V + 1), cudaMemcpyHostToDevice));
    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_E, E, sizeof(int) * len_E, cudaMemcpyHostToDevice));
    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_sources, sources, sizeof(int) * len_sources, cudaMemcpyHostToDevice));
    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_W, W, sizeof(double) * len_E, cudaMemcpyHostToDevice));

    d_calc_min_edge<<<dijkstra_grid_size, dijkstra_block_size>>>(d_V, d_E, d_W, len_V, len_E, d_min_edge);

    d_sssp_dijkstra<<<min_edge_grid_size, min_edge_block_size>>>(d_curr_node ,d_V, d_E, d_W, d_min_edge, d_sources, 
                                    d_dist_2D, d_U_2D, d_F_2D, len_V, len_E, len_sources, target, warp_size);

    EXIT_IF_CUDA_FAILED(cudaMemcpy(res, d_dist_2D, sizeof(double) * len_sources * len_V, cudaMemcpyDeviceToHost));

exit:
    cudaFree(d_curr_node);
    cudaFree(d_V);
    cudaFree(d_E);
    cudaFree(d_sources);
    cudaFree(d_U_2D);
    cudaFree(d_F_2D);
    cudaFree(d_W);
    cudaFree(d_min_edge);
    cudaFree(d_dist_2D);

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

} // namespace gpu_easygraph