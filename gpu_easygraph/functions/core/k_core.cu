#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#include "common.h"

static __global__ void d_calc_deg(
    _IN_ int* d_V,
    _IN_ int* d_E,
    _IN_ int len_V,
    _IN_ int len_E,
    _OUT_ int* d_deg
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= len_V) {
        return;
    }
    d_deg[tid] = d_V[tid + 1] - d_V[tid];
}



static __global__ void d_k_core_scan_local(
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



static __global__ void d_k_core_loop_local(
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


int cuda_k_core_local (
    _IN_ int* V,
    _IN_ int* E,
    _IN_ int len_V,
    _IN_ int len_E,
    _OUT_ int* k_core_res
)
{
    int cuda_ret = cudaSuccess;
    int EG_ret = EG_GPU_SUCC;

    int block_size = 256;
    int grid_size = 56; //TODO

    int count = 0, level = 0;

    int *d_V = NULL, *d_E = NULL, *d_deg = NULL, *d_k_core_res = NULL,
            *d_buf_2D = NULL, *d_buf_tail_2D = NULL, *d_count = NULL;

    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_V, sizeof(int) * (len_V + 1)));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_E, sizeof(int) * len_E));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_deg, sizeof(int) * len_V));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_k_core_res, sizeof(int) * len_V));
    // TMP TODO TODO size
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_buf_2D, sizeof(int) * grid_size * len_V));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_buf_tail_2D, sizeof(int) * grid_size));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_count, sizeof(int)));

    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_V, V, sizeof(int) * (len_V + 1), cudaMemcpyHostToDevice));
    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_E, E, sizeof(int) * len_E, cudaMemcpyHostToDevice));

    EXIT_IF_CUDA_FAILED(cudaMemset(d_count, 0, sizeof(int)));

    d_calc_deg<<<len_V / block_size + 1, block_size>>>(d_V, d_E, len_V, len_E, d_deg);
    while (count < len_V) {
        EXIT_IF_CUDA_FAILED(cudaMemset(d_buf_tail_2D, 0, sizeof(int) * grid_size));

        d_k_core_scan_local<<<grid_size, block_size>>>(d_deg, len_V, level, d_buf_2D, d_buf_tail_2D);

        d_k_core_loop_local<<<grid_size, block_size>>>(d_V, d_E, d_deg, len_V, len_E, level,
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



static __global__ void d_k_core_scan_global(
    _IN_ int* d_deg,
    _IN_ int len_V,
    _IN_ int level,
    _IN_ int* d_buf,
    _OUT_ int* d_buf_rear
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int tnum = blockDim.x * gridDim.x;

    for (int v = tid; v < len_V; v += tnum) {
        if (d_deg[v] == level) {
            int buf_idx = atomicAdd(d_buf_rear, 1);
            d_buf[buf_idx] = v;
        }
    }
}



static __global__ void d_k_core_loop_global(
    _IN_ int* d_V,
    _IN_ int* d_E,
    _OUT_ int* d_deg,
    _IN_ int len_V,
    _IN_ int len_E,
    _IN_ int level,
    _BUFFER_ int* d_buf,
    _IN_ int buf_front,
    _IN_ int buf_rear,
    _BUFFER_ int* d_buf_next_rear
)
{
    int warp_size = 32;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp_id = tid / warp_size;
    const int lane_id = tid % warp_size;
    const int warp_num = gridDim.x * blockDim.x / warp_size;

    for (int v_idx = buf_front + warp_id; v_idx < buf_rear; v_idx += warp_num) {
        int v = d_buf[v_idx];
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
                    int loc = atomicAdd(d_buf_next_rear, 1);
                    d_buf[loc] = nbr;
                }

                if (a <= level) {
                    atomicAdd(d_deg + nbr, 1);
                }
            }
        }
    }
}


int cuda_k_core_global (
    _IN_ int* V,
    _IN_ int* E,
    _IN_ int len_V,
    _IN_ int len_E,
    _OUT_ int* k_core_res
)
{
    int cuda_ret = cudaSuccess;
    int EG_ret = EG_GPU_SUCC;

    int block_size = 256;
    int grid_size = 56;

    int count = 0, level = 0;
    int buf_front = 0, buf_rear = 0, buf_next_rear = 0;

    int *d_V = NULL, *d_E = NULL, *d_deg = NULL, *d_k_core_res = NULL,
            *d_buf = NULL, *d_buf_rear = NULL, *d_buf_next_rear = NULL;

    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_V, sizeof(int) * len_V));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_E, sizeof(int) * len_E));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_deg, sizeof(int) * len_V));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_k_core_res, sizeof(int) * len_V));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_buf, sizeof(int) * len_V));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_buf_rear, sizeof(int)));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_buf_next_rear, sizeof(int)));

    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_V, V, sizeof(int) * len_V, cudaMemcpyHostToDevice));
    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_E, E, sizeof(int) * len_E, cudaMemcpyHostToDevice));

    d_calc_deg<<<len_V / block_size + 1, block_size>>>(d_V, d_E, len_V, len_E, d_deg);
    while (count < len_V) {
        EXIT_IF_CUDA_FAILED(cudaMemset(d_buf_rear, 0, sizeof(int)));

        d_k_core_scan_global<<<grid_size, block_size>>>(d_deg, len_V, level, d_buf, d_buf_rear);

        buf_front = 0;
        EXIT_IF_CUDA_FAILED(cudaMemcpy(&buf_rear, d_buf_rear, sizeof(int), cudaMemcpyDeviceToHost));
        EXIT_IF_CUDA_FAILED(cudaMemcpy(d_buf_next_rear, d_buf_rear, sizeof(int), cudaMemcpyDeviceToDevice));

        while (1) {
            d_k_core_loop_global<<<grid_size, block_size>>>(d_V, d_E, d_deg, len_V, len_E, level,
                                                    d_buf, buf_front, buf_rear, d_buf_next_rear);

            EXIT_IF_CUDA_FAILED(cudaMemcpy(&buf_next_rear, d_buf_next_rear, sizeof(int), cudaMemcpyDeviceToHost));
            if (buf_rear == buf_next_rear) {
                break;
            }

            buf_front = buf_rear;
            buf_rear = buf_next_rear;
        }

        count += buf_next_rear;

        ++level;
    }

    EXIT_IF_CUDA_FAILED(cudaMemcpy(k_core_res, d_deg, sizeof(int) * len_V, cudaMemcpyDeviceToHost));

exit:
    cudaFree(d_V);
    cudaFree(d_E);
    cudaFree(d_deg);
    cudaFree(d_k_core_res);
    cudaFree(d_buf);
    cudaFree(d_buf_rear);
    cudaFree(d_buf_next_rear);

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



int cuda_k_core (
    _IN_ int* V,
    _IN_ int* E,
    _IN_ int len_V,
    _IN_ int len_E,
    _OUT_ int* k_core_res
)
{
    return cuda_k_core_local(V, E, len_V, len_E, k_core_res);
}
