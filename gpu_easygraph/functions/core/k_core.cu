#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#include "common.h"

namespace gpu_easygraph {

static __global__ void d_calc_deg(
    _IN_ int* d_V,
    _IN_ int* d_E,
    _IN_ int len_V,
    _IN_ int len_E,
    _OUT_ int* d_deg
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int tnum = blockDim.x * gridDim.x;
    for (int u = tid; u < len_V; u += tnum) {
        d_deg[u] = d_V[u + 1] - d_V[u];
    }
}



static __global__ void d_k_core_scan(
    _IN_ int* d_deg,
    _IN_ int len_V,
    _IN_ int level,
    _IN_ int* d_buf_2D,
    _IN_ int* d_buf_tail_2D
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int threads_num = blockDim.x * gridDim.x;
    int* d_buf = d_buf_2D + blockIdx.x * len_V;
    __shared__ int buf_tail;

    if (threadIdx.x == 0) {
        buf_tail = 0;
    }
    __syncthreads();

    for (int base = 0; base < len_V; base += threads_num) {
        int v = base + tid;

        if (v >= len_V) {
            continue;
        }

        if (d_deg[v] == level) {
            int buf_idx = atomicAdd(&buf_tail, 1);
            d_buf[buf_idx] = v;
        }

    }
    __syncthreads();    

    if (threadIdx.x == 0) {
        d_buf_tail_2D[blockIdx.x] = buf_tail;
    }
}



static __global__ void d_k_core_loop(
    _IN_ int* d_V,
    _IN_ int* d_E,
    _OUT_ int* d_deg,
    _IN_ int len_V,
    _IN_ int len_E,
    _IN_ int level,
    _IN_ int* d_buf_2D,
    _IN_ int* d_buf_tail_2D,
    _OUT_ int* d_count
)
{
    int warp_size = 32;
    int tid = threadIdx.x;
    int* d_buf = d_buf_2D + blockIdx.x * len_V;
    int warp_id = tid / warp_size;
    int lane_id = tid % warp_size;
    int reg_tail;
    int i;

    __shared__ int buf_tail;
    __shared__ int base;

    if (threadIdx.x == 0) {
        base = 0;
        buf_tail = d_buf_tail_2D[blockIdx.x];
    }
    __syncthreads();

    while (1) {
        __syncthreads();

        if (base == buf_tail) {
            break;
        }

        i = base + warp_id;
        reg_tail = buf_tail;
        __syncthreads();

        if (i >= reg_tail) {
            continue;
        }

        if (threadIdx.x == 0) {
            base += blockDim.x / warp_size;
            if (reg_tail < base) {
                base = reg_tail;
            }
        }

        int v = d_buf[i];
        int edge_start = d_V[v];
        int edge_end = d_V[v + 1];

        while (1) {
            __syncwarp();

            if (edge_start >= edge_end) {
                break;
            }

            int curr_e = edge_start + lane_id;
            edge_start += warp_size;

            if (curr_e >= edge_end) {
                continue;
            }

            int nbr = d_E[curr_e];
            if (d_deg[nbr] > level) {
                int a = atomicSub(d_deg + nbr, 1);

                if (a == level + 1) {
                    int loc = atomicAdd(&buf_tail, 1);
                    d_buf[loc] = nbr;
                }

                if (a <= level) {
                    atomicAdd(d_deg + nbr, 1);
                }
            }
        }
    }

    if (threadIdx.x == 0 && buf_tail) {
        atomicAdd(d_count, buf_tail);
    }
    
}


int cuda_k_core (
    _IN_ const int* V,
    _IN_ const int* E,
    _IN_ int len_V,
    _IN_ int len_E,
    _OUT_ int* k_core_res
)
{
    int cuda_ret = cudaSuccess;
    int EG_ret = EG_GPU_SUCC;

    int calc_deg_block_size;
    int calc_deg_grid_size;
    int scan_block_size;
    int scan_grid_size;
    int loop_block_size;
    int loop_grid_size;

    cudaOccupancyMaxPotentialBlockSize(&calc_deg_grid_size, &calc_deg_block_size, d_calc_deg, 0, 0); 
    cudaOccupancyMaxPotentialBlockSize(&scan_grid_size, &scan_block_size, d_k_core_scan, 0, 0); 
    cudaOccupancyMaxPotentialBlockSize(&loop_grid_size, &loop_block_size, d_k_core_loop, 0, 0); 

    int k_core_grid_size = max(scan_grid_size, loop_grid_size);

    int count = 0, level = 0;

    int *d_V = NULL, *d_E = NULL, *d_deg = NULL, *d_k_core_res = NULL,
            *d_buf_2D = NULL, *d_buf_tail_2D = NULL, *d_count = NULL;

    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_V, sizeof(int) * (len_V + 1)));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_E, sizeof(int) * len_E));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_deg, sizeof(int) * len_V));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_k_core_res, sizeof(int) * len_V));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_buf_2D, sizeof(int) * k_core_grid_size * len_V));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_buf_tail_2D, sizeof(int) * k_core_grid_size));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_count, sizeof(int)));

    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_V, V, sizeof(int) * (len_V + 1), cudaMemcpyHostToDevice));
    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_E, E, sizeof(int) * len_E, cudaMemcpyHostToDevice));

    EXIT_IF_CUDA_FAILED(cudaMemset(d_count, 0, sizeof(int)));

    d_calc_deg<<<calc_deg_grid_size, calc_deg_block_size>>>(d_V, d_E, len_V, len_E, d_deg);
    while (count < len_V) {
        EXIT_IF_CUDA_FAILED(cudaMemset(d_buf_tail_2D, 0, sizeof(int) * k_core_grid_size));

        d_k_core_scan<<<k_core_grid_size, scan_block_size>>>(d_deg, len_V, level, d_buf_2D, d_buf_tail_2D);

        d_k_core_loop<<<k_core_grid_size, loop_block_size>>>(d_V, d_E, d_deg, len_V, len_E, level,
                                                    d_buf_2D, d_buf_tail_2D, d_count);
        
        EXIT_IF_CUDA_FAILED(cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost));

        ++level;
    }

    EXIT_IF_CUDA_FAILED(cudaMemcpy(k_core_res, d_deg, sizeof(int) * len_V, cudaMemcpyDeviceToHost));

exit:
    cudaFree(d_V);
    cudaFree(d_E);
    cudaFree(d_deg);
    cudaFree(d_k_core_res);
    cudaFree(d_buf_2D);
    cudaFree(d_buf_tail_2D);
    cudaFree(d_count);

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