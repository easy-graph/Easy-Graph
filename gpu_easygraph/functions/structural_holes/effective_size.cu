#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#include "common.h"
#define NODES_PER_BLOCK 1

namespace gpu_easygraph {

enum norm_t { SUM = 0, MAX = 1 };

static __device__ double mutual_weight(
    const int* V,
    const int* E,
    const double* W,
    int u,
    int v
) {
    double a_uv = 0.0;
    for (int i = V[u]; i < V[u+1]; i++) {
        if (E[i] == v) {
            a_uv = W[i];
            break;
        }
    }
    return a_uv;
}

static __device__ double normalized_mutual_weight(
    const int* V,
    const int* E,
    const double* W, 
    int u,
    int v,
    norm_t norm
) {
    double weight_uv = mutual_weight(V, E, W, u, v);

    double scale = 0.0;
    if (norm == SUM) {
        for (int i = V[u]; i < V[u+1]; i++) {
            int neighbor = E[i];
            double weight_uw = mutual_weight(V, E, W, u, neighbor);
            scale += weight_uw;
        }
    } else if (norm == MAX) {
        for (int i = V[u]; i < V[u+1]; i++) {
            int neighbor = E[i];
            double weight_uw = mutual_weight(V, E, W, u, neighbor);
            scale = fmax(scale,weight_uw);
        }
    }
    return (scale==0.0) ? 0.0 : (weight_uv / scale);
}

static __device__ double directed_mutual_weight(
    const int* V,
    const int* E,
    const double* W,
    int u,
    int v
) {
    double a_uv = 0.0, a_vu = 0.0;
    for (int i = V[u]; i < V[u+1]; i++) {
        if (E[i] == v) {
            a_uv = W[i];
            break;
        }
    }
    for (int i = V[v]; i < V[v+1]; i++) {
        if (E[i] == u) {
            a_vu = W[i];
            break;
        }
    }
    return a_uv + a_vu;
}

static __device__ double directed_normalized_mutual_weight(
    const int* V,
    const int* E,
    const int* row, 
    const int* col, 
    const double* W, 
    int num_edges,
    int u,
    int v,
    norm_t norm
) {
    double weight_uv = directed_mutual_weight(V, E, W, u, v);

    double scale = 0.0;
    if(norm==SUM){
        for (int i = V[u]; i < V[u+1]; i++) {
            int neighbor = E[i];
            double weight_uw = directed_mutual_weight(V, E, W, u, neighbor);
            scale += weight_uw;
        }

        for (int i = 0; i < num_edges; i++) {
            if (col[i] == u) {
                int neighbor = row[i];
                double weight_wu = directed_mutual_weight(V, E, W, u, neighbor);
                scale += weight_wu;
            }
        }
    }else if(norm==MAX){
        for (int i = V[u]; i < V[u+1]; i++) {
            int neighbor = E[i];
            double weight_uw = directed_mutual_weight(V, E, W, u, neighbor);
            scale = fmax(scale,weight_uw);
        }

        for (int i = 0; i < num_edges; i++) {
            if (col[i] == u) {
                int neighbor = row[i];
                double weight_wu = directed_mutual_weight(V, E, W, u, neighbor);
                scale = fmax(scale,weight_wu);
            }
        }
    }
    return (scale==0.0) ? 0.0 : (weight_uv / scale);
}

static __device__ double redundancy(
    const int* V,
    const int* E,
    const double* W,
    const int num_nodes,
    int u,
    int v
) {
    double r = 0.0;
    for (int i = V[v]; i < V[v + 1]; i++) {
        int w = E[i];
        r += normalized_mutual_weight(V, E, W, u, w, SUM) * normalized_mutual_weight(V, E, W, v, w, MAX);
    }
    return 1-r;
}


__inline__ __device__ double warp_reduce_sum(double val)
{
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
    {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__inline__ __device__ double block_reduce_sum(double val)
{
    val = warp_reduce_sum(val);

    __shared__ double shared[32];
    int warp_id = threadIdx.x / warpSize;
    if (threadIdx.x % warpSize == 0)
    {
        shared[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0)
    {
        val = (threadIdx.x < (blockDim.x / warpSize)) ? shared[threadIdx.x] : 0.0;
        val = warp_reduce_sum(val);
    }
    return val;
}

__global__ void calculate_effective_size(
    const int* __restrict__ V,
    const int* __restrict__ E,
    const double* __restrict__ W,
    const int num_nodes,
    const int* __restrict__ node_mask,
    double* __restrict__ effective_size_results
) {
    int u = blockIdx.x;
    if (u >= num_nodes || !node_mask[u]) return;

    int neighbor_start = V[u];
    int neighbor_end = V[u + 1];
    int degree = neighbor_end - neighbor_start;

    int threads_per_block = blockDim.x;

    double redundancy_sum = 0.0;
    for (int idx = threadIdx.x; idx < degree; idx += threads_per_block) {
        int i = neighbor_start + idx;
        int v = E[i];
        if (v != u) {
            double r = 0.0;
            for (int j = V[v]; j < V[v + 1]; j++) {
                int w = E[j];
                r += normalized_mutual_weight(V, E, W, u, w, SUM) * 
                     normalized_mutual_weight(V, E, W, v, w, MAX);
            }
            redundancy_sum += 1 - r;
        }
    }

    redundancy_sum = block_reduce_sum(redundancy_sum);

    if (threadIdx.x == 0) {
        effective_size_results[u] = redundancy_sum;
    }
}

static __device__ double directed_redundancy(
    const int* V,
    const int* E,
    const int* row,
    const int* col,
    const double* W,
    const int num_nodes,
    const int num_edges,
    int u,
    int v
) {
    double r = 0.0;
    for (int i = V[v]; i < V[v + 1]; i++) {
        int w = E[i];
        r += directed_normalized_mutual_weight(V, E, row,col,W,num_edges, u, w,SUM) * directed_normalized_mutual_weight(V, E, row,col,W, num_edges, v,w,MAX);
    }
    for (int i = 0; i < num_edges; i++) {
        if (col[i] == v) {
            int w = row[i];
            r += directed_normalized_mutual_weight(V, E, row,col,W,num_edges, u, w,SUM) * directed_normalized_mutual_weight(V, E, row,col,W, num_edges, v,w,MAX);
        }
    }
    return 1-r;
}

__global__ void directed_calculate_effective_size(
    const int* V,
    const int* E,
    const int* row,
    const int* col,
    const double* W, 
    const int num_nodes,
    const int num_edges,
    const int* node_mask,
    double* effective_size_results
) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= num_nodes || !node_mask[u]) return;
    double redundancy_sum = 0.0;
    bool is_nan = true;

    for (int i = V[u]; i < V[u + 1]; i++) {
        int v = E[i];
        if (v == u) continue;
        is_nan = false;
        redundancy_sum += directed_redundancy(V,E,row,col,W,num_nodes,num_edges,u,v);
    }
    for (int i = 0; i < num_edges; i++) {
        if (col[i] == u) {
            int v = row[i];
            redundancy_sum += directed_redundancy(V,E,row,col,W,num_nodes,num_edges,u,v);
        }
    }
    effective_size_results[u] = is_nan ? NAN : redundancy_sum;
}


int cuda_effective_size(
    _IN_ const int* V,
    _IN_ const int* E,
    _IN_ const int* row,
    _IN_ const int* col,
    _IN_ const double* W,
    _IN_ int num_nodes,
    _IN_ int num_edges,
    _IN_ bool is_directed,
    _IN_ int* node_mask,
    _OUT_ double* effective_size_results
) {
    int cuda_ret = cudaSuccess;
    int EG_ret = EG_GPU_SUCC;
    int min_grid_size = 0;
    int block_size = 0;

    
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, calculate_effective_size, 0, 0);

    int grid_size = (num_nodes + block_size * NODES_PER_BLOCK - 1) / (block_size * NODES_PER_BLOCK);

    int* d_V;
    int* d_E;
    int* d_row;
    int* d_col;
    double* d_W;
    int* d_node_mask;
    double* d_effective_size_results;

    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_V, (num_nodes+1) * sizeof(int)));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_E, num_edges * sizeof(int)));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_row, num_edges * sizeof(int)));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_col, num_edges * sizeof(int)));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_W, num_edges * sizeof(double)));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_node_mask, num_nodes * sizeof(int)));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_effective_size_results, num_nodes * sizeof(double)));

    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_V, V, (num_nodes+1) * sizeof(int), cudaMemcpyHostToDevice));
    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_E, E, num_edges * sizeof(int), cudaMemcpyHostToDevice));
    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_row, row, num_edges * sizeof(int), cudaMemcpyHostToDevice));
    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_col, col, num_edges * sizeof(int), cudaMemcpyHostToDevice));
    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_node_mask, node_mask, num_nodes * sizeof(int), cudaMemcpyHostToDevice));
    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_W, W, num_edges * sizeof(double), cudaMemcpyHostToDevice));

    if(is_directed){
        directed_calculate_effective_size<<<grid_size, block_size>>>(d_V, d_E, d_row, d_col, d_W, num_nodes, num_edges, d_node_mask, d_effective_size_results);
    }else{
        int block_size = 256; 
        int grid_size = (num_nodes + NODES_PER_BLOCK - 1) / NODES_PER_BLOCK;
        calculate_effective_size<<<grid_size, block_size>>>(d_V, d_E, d_W, num_nodes, d_node_mask, d_effective_size_results);
        
    }

    EXIT_IF_CUDA_FAILED(cudaMemcpy(effective_size_results, d_effective_size_results, num_nodes * sizeof(double), cudaMemcpyDeviceToHost));

exit:
    cudaFree(d_V);
    cudaFree(d_E);
    cudaFree(d_row);
    cudaFree(d_col);
    cudaFree(d_W);
    cudaFree(d_node_mask);
    cudaFree(d_effective_size_results);

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