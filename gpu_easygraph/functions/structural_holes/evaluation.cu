#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#include "common.h"

namespace gpu_easygraph {

static __device__ double mutual_weight(
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
    return a_uv + a_vu;
}

static __device__ double normalized_mutual_weight(
    const int* V,
    const int* E,
    const double* W, 
    int u,
    int v
) {
    double weight_uv = mutual_weight(V, E, W, u, v);

    double scale = 0.0;
    for (int i = V[u]; i < V[u+1]; i++) {
        int neighbor = E[i];
        double weight_uw = mutual_weight(V, E, W, u, neighbor);
        scale += weight_uw;
    }
    return (scale==0.0) ? 0.0 : (weight_uv / scale);
}

static __device__ double local_constraint(
    const int* V,
    const int* E,
    const double* W,
    int u,
    int v
) {
    double direct = normalized_mutual_weight(V,E,W,u,v);
    double indirect = 0.0;
    for (int i = V[u]; i < V[u+1]; i++) {
        int neighbor = E[i];
        double norm_uw = normalized_mutual_weight(V, E, W, u, neighbor);
        double norm_wv = normalized_mutual_weight(V, E, W, neighbor, v);
        indirect += norm_uw * norm_wv;
    }
    double local_constraint_of_uv = (direct + indirect) * (direct + indirect);
    return local_constraint_of_uv;
}

__global__ void calculate_constraints(
    const int* V,
    const int* E,
    const double* W, 
    const int num_nodes, 
    double* constraint_results
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_nodes) return;
    double constraint_of_v = 0.0;
    bool is_nan = true;
    for (int i = V[v]; i < V[v+1]; i++) {
        is_nan = false;
        int neighbor = E[i]; 

        constraint_of_v += local_constraint(V, E, W, v, neighbor);
    }
    constraint_results[v] = (is_nan) ? NAN : constraint_of_v;
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
    int v
) {
    double weight_uv = directed_mutual_weight(V, E, W, u, v);

    double scale = 0.0;
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
    return (scale==0.0) ? 0.0 : (weight_uv / scale);
}

static __device__ double directed_local_constraint(
    const int* V,
    const int* E,
    const int* row, 
    const int* col, 
    const double* W,
    int num_edges,
    int u,
    int v
) {
    double direct = directed_normalized_mutual_weight(V,E,row,col,W,num_edges,u,v);
    double indirect = 0.0;
    for (int i = V[u]; i < V[u+1]; i++) {
        int neighbor = E[i];
        double norm_uw = directed_normalized_mutual_weight(V, E, row, col, W, num_edges, u, neighbor);
        double norm_wv = directed_normalized_mutual_weight(V, E, row, col, W, num_edges, neighbor, v);
        indirect += norm_uw * norm_wv;
    }

    for (int i = 0; i < num_edges; i++) {
        if (col[i] == u) {
            int neighbor = row[i];
            double norm_uw = directed_normalized_mutual_weight(V, E, row, col, W, num_edges, u, neighbor);
            double norm_wv = directed_normalized_mutual_weight(V, E, row, col, W, num_edges, neighbor, v);
            indirect += norm_uw * norm_wv;
        }
    }
    double local_constraint_of_uv = (direct + indirect) * (direct + indirect);
    return local_constraint_of_uv;
}

__global__ void directed_calculate_constraints(
    const int* V,
    const int* E,
    const int* row, 
    const int* col, 
    const double* W,  
    int num_nodes,
    int num_edges,
    double* constraint_results
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_nodes) return;
    double constraint_of_v = 0.0;
    bool is_nan = true;
    for (int i = V[v]; i < V[v+1]; i++) {
        is_nan = false;
        int neighbor = E[i]; 
        constraint_of_v += directed_local_constraint(V, E, row, col, W, num_edges, v, neighbor);
    }
    for (int i = 0; i < num_edges; i++) {
        if (col[i] == v) {
            int neighbor = row[i];
            constraint_of_v += directed_local_constraint(V, E, row, col, W, num_edges, v, neighbor);
        }
    }
    constraint_results[v] = (is_nan) ? NAN : constraint_of_v;
}

int cuda_constraint(
    _IN_ const int* V,
    _IN_ const int* E,
    _IN_ const int* row,
    _IN_ const int* col,
    _IN_ const double* W,
    _IN_ int num_nodes,
    _IN_ int num_edges,
    _IN_ bool is_directed,
    _OUT_ double* constraint_results
) {
    int cuda_ret = cudaSuccess;
    int EG_ret = EG_GPU_SUCC;
    int min_grid_size = 0;
    int block_size = 0;
    
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, calculate_constraints, 0, 0);
    int grid_size = (num_nodes + block_size - 1) / block_size;
    
    int* d_V;
    int* d_E;
    int* d_row;
    int* d_col;
    double* d_W;
    double* d_constraint_results;

    // 分配CUDA设备内存
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_V, (num_nodes+1) * sizeof(int)));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_E, num_edges * sizeof(int)));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_row, num_edges * sizeof(int)));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_col, num_edges * sizeof(int)));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_W, num_edges * sizeof(double)));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_constraint_results, num_nodes * sizeof(double)));

    // 将数据从主机拷贝到设备
    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_V, V, (num_nodes+1) * sizeof(int), cudaMemcpyHostToDevice));
    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_E, E, num_edges * sizeof(int), cudaMemcpyHostToDevice));
    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_row, row, num_edges * sizeof(int), cudaMemcpyHostToDevice));
    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_col, col, num_edges * sizeof(int), cudaMemcpyHostToDevice));
    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_W, W, num_edges * sizeof(double), cudaMemcpyHostToDevice));


    // 调用 CUDA 内核
    if(is_directed){
        directed_calculate_constraints<<<grid_size, block_size>>>(d_V, d_E, d_row, d_col, d_W, num_nodes, num_edges, d_constraint_results);
    }else{
        calculate_constraints<<<grid_size, block_size>>>(d_V, d_E, d_W, num_nodes, d_constraint_results);
    }

    EXIT_IF_CUDA_FAILED(cudaMemcpy(constraint_results, d_constraint_results, num_nodes * sizeof(double), cudaMemcpyDeviceToHost));
exit:

    cudaFree(d_V);
    cudaFree(d_E);
    cudaFree(d_row);
    cudaFree(d_col);
    cudaFree(d_W);
    cudaFree(d_constraint_results);
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