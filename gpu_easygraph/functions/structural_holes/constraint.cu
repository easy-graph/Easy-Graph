#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#include "common.h"
// #define NODES_PER_BLOCK 1

namespace gpu_easygraph {

enum norm_t { SUM = 0, MAX = 1 };

// static __device__ double mutual_weight(
//     const int* V,
//     const int* E,
//     const double* W,
//     int u,
//     int v
// ) {
//     double a_uv = 0.0;
//     for (int i = V[u]; i < V[u+1]; i++) {
//         if (E[i] == v) {
//             a_uv = W[i];
//             break;
//         }
//     }
//     return a_uv;
// }

// static __device__ double normalized_mutual_weight(
//     const int* V,
//     const int* E,
//     const double* W, 
//     int u,
//     int v,
//     norm_t norm
// ) {
//     double weight_uv = mutual_weight(V, E, W, u, v);

//     double scale = 0.0;
//     if (norm == SUM) {
//         for (int i = V[u]; i < V[u+1]; i++) {
//             int neighbor = E[i];
//             double weight_uw = mutual_weight(V, E, W, u, neighbor);
//             scale += weight_uw;
//         }
//     } else if (norm == MAX) {
//         for (int i = V[u]; i < V[u+1]; i++) {
//             int neighbor = E[i];
//             double weight_uw = mutual_weight(V, E, W, u, neighbor);
//             scale = fmax(scale,weight_uw);
//         }
//     }
//     return (scale==0.0) ? 0.0 : (weight_uv / scale);
// }

// static __device__ double local_constraint(
//     const int* V,
//     const int* E,
//     const double* W,
//     int u,
//     int v
// ) {
//     double direct = normalized_mutual_weight(V,E,W,u,v,SUM);
//     double indirect = 0.0;
//     for (int i = V[u]; i < V[u+1]; i++) {
//         int neighbor = E[i];
//         double norm_uw = normalized_mutual_weight(V, E, W, u, neighbor,SUM);
//         double norm_wv = normalized_mutual_weight(V, E, W, neighbor, v,SUM);
//         indirect += norm_uw * norm_wv;
//     }
//     double local_constraint_of_uv = (direct + indirect) * (direct + indirect);
//     return local_constraint_of_uv;
// }

// __global__ void calculate_constraints(
//     const int* __restrict__ V,
//     const int* __restrict__ E,
//     const double* __restrict__ W, 
//     const int num_nodes, 
//     const int* __restrict__ node_mask,
//     double* __restrict__ constraint_results
// ) {
//     int start_node = blockIdx.x * NODES_PER_BLOCK;
//     int end_node = min(start_node + NODES_PER_BLOCK, num_nodes);

//     for (int v = start_node; v < end_node; ++v) {
//         if (!node_mask[v]) continue;

//         double constraint_of_v = 0.0;
//         bool is_nan = true;

//         __shared__ double shared_constraint[256];
//         double local_sum = 0.0;

//         for (int i = V[v] + threadIdx.x; i < V[v + 1]; i += blockDim.x) {
//             is_nan = false;
//             int neighbor = E[i];
//             local_sum += local_constraint(V, E, W, v, neighbor);
//         }

//         shared_constraint[threadIdx.x] = local_sum;
//         __syncthreads();

//         for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
//             if (threadIdx.x < offset) {
//                 shared_constraint[threadIdx.x] += shared_constraint[threadIdx.x + offset];
//             }
//             __syncthreads();
//         }

//         if (threadIdx.x == 0) {
//             constraint_results[v] = (is_nan) ? NAN : shared_constraint[0];
//         }
//     }
// }

__device__ double mutual_weight(
    const int* rowPtrOut,
    const int* colIdxOut,
    const double* valOut,
    const int* rowPtrIn,
    const int* colIdxIn,
    const double* valIn,
    int u,
    int v
)
{
    double w_uv = 0.0, w_vu = 0.0;
    for (int i = rowPtrOut[u]; i < rowPtrOut[u+1]; i++) {
        if (colIdxOut[i] == v) {
            w_uv = valOut[i];
            break;
        }
    }
    for (int i = rowPtrIn[u]; i < rowPtrIn[u+1]; i++) {
        if (colIdxIn[i] == v) {
            w_vu = valIn[i];
            break;
        }
    }
    // printf("u=%d, v=%d, w_uv=%f, w_vu=%f\n", u, v, w_uv, w_vu);
    return w_uv + w_vu;
}

__global__ void compute_out_in_sum(
    const int* rowPtrOut,
    const double* valOut,
    const int* rowPtrIn,
    const double* valIn,
    int num_nodes,
    const int* node_mask,
    double* d_sum
)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= num_nodes || !node_mask[u]) return; // 跳过未标记的节点

    double sum_val = 0.0;

    for (int i = rowPtrOut[u]; i < rowPtrOut[u + 1]; i++) {
        sum_val += valOut[i];
    }

    for (int i = rowPtrIn[u]; i < rowPtrIn[u + 1]; i++) {
        sum_val += valIn[i];
    }
    d_sum[u] = sum_val;
    // printf("Node %d: sum_val = %f\n", u, sum_val);
}

__device__ double local_constraint(
    const int* rowPtrOut,
    const int* colIdxOut,
    const double* valOut,
    const int* rowPtrIn,
    const int* colIdxIn,
    const double* valIn,
    const double* d_sum,
    int u,
    int v
) {
    double weight_uv = mutual_weight(rowPtrOut, colIdxOut, valOut, rowPtrIn, colIdxIn, valIn, u, v);
    double sum_u = 0.0;
    for (int i = rowPtrOut[u]; i < rowPtrOut[u+1]; i++) {
        sum_u += valOut[i];
    }
    for (int i = rowPtrIn[u]; i < rowPtrIn[u+1]; i++) {
        sum_u += valIn[i];
    }
    double direct = (sum_u == 0.0) ? 0.0 : (weight_uv / sum_u);

    double indirect = 0.0;
    double scale_u = d_sum[u];
    for (int i = rowPtrOut[u]; i < rowPtrOut[u+1]; i++) {
        int nbr = colIdxOut[i];
        double w_uw = mutual_weight(rowPtrOut, colIdxOut, valOut, rowPtrIn, colIdxIn, valIn, u, nbr);
        double w_wv = mutual_weight(rowPtrOut, colIdxOut, valOut, rowPtrIn, colIdxIn, valIn, nbr, v);

        // double scale_u = d_sum[u];   // u的分母
        double scale_nbr = d_sum[nbr];  // neighbor的分母

        double norm_uw = (scale_u == 0.0) ? 0.0 : (w_uw / scale_u);
        double norm_wv = (scale_nbr == 0.0) ? 0.0 : (w_wv / scale_nbr);

        indirect += norm_uw * norm_wv;
    }
    for (int i = rowPtrIn[u]; i < rowPtrIn[u+1]; i++) {
        int nbr = colIdxIn[i];
        double w_uw = mutual_weight(rowPtrOut, colIdxOut, valOut, rowPtrIn, colIdxIn, valIn, u, nbr);
        double w_wv = mutual_weight(rowPtrOut, colIdxOut, valOut, rowPtrIn, colIdxIn, valIn, nbr, v);

        
        double scale_nbr = d_sum[nbr];

        double norm_uw = (scale_u == 0.0) ? 0.0 : (w_uw / scale_u);
        double norm_wv = (scale_nbr == 0.0) ? 0.0 : (w_wv / scale_nbr);

        indirect += norm_uw * norm_wv;
    }
    double result = (direct + indirect) * (direct + indirect);
    printf("u=%d, v=%d, direct=%f, indirect=%f\n", u, v, direct, indirect);
    return result;
}

__global__ void calculate_constraints(
    int num_nodes,
    int NODES_PER_BLOCK,
    const int* rowPtrOut,
    const int* colIdxOut,
    const double* valOut,
    const int* rowPtrIn,
    const int* colIdxIn,
    const double* valIn,
    const double* d_sum,
    const int* node_mask,
    double* constraint_results
){
    int start_node = blockIdx.x * NODES_PER_BLOCK;
    int end_node = min(start_node + NODES_PER_BLOCK, num_nodes);

    for (int v = start_node; v < end_node; ++v) {
        if (!node_mask[v]) continue;
        bool is_nan = true;
        // bool is_nan = false;
        __shared__ double shared_constraint[256];
        double local_sum = 0.0;

        for (int i = rowPtrOut[v] + threadIdx.x; i < rowPtrOut[v + 1]; i += blockDim.x) {
            is_nan = false;
            int neighbor = colIdxOut[i];
            local_sum += local_constraint(rowPtrOut, colIdxOut, valOut, rowPtrIn, colIdxIn, valIn, d_sum, v, neighbor);
        }

        for (int i = rowPtrIn[v] + threadIdx.x; i < rowPtrIn[v + 1]; i += blockDim.x) {
            int neighbor = colIdxIn[i];
            local_sum += local_constraint(rowPtrOut, colIdxOut, valOut, rowPtrIn, colIdxIn, valIn, d_sum, v, neighbor);
        }

        shared_constraint[threadIdx.x] = local_sum;
        __syncthreads();

        for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
            if (threadIdx.x < offset) {
                shared_constraint[threadIdx.x] += shared_constraint[threadIdx.x + offset];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            constraint_results[v] = (is_nan) ? NAN : shared_constraint[0];
        }
    }
}

int cuda_constraint(
    _IN_ int num_nodes,
    _IN_ int len_rowPtrOut,
    _IN_ int len_colIdxOut,
    _IN_ const int* rowPtrOut,
    _IN_ const int* colIdxOut,
    _IN_ const double* valOut,
    _IN_ const int* rowPtrIn,
    _IN_ const int* colIdxIn,
    _IN_ const double* valIn,
    _IN_ bool is_directed,
    _IN_ int* node_mask,
    _OUT_ double* constraints
) {
    int cuda_ret = cudaSuccess;
    int EG_ret = EG_GPU_SUCC;
    
    int* d_rowPtrOut = nullptr;
    int* d_colIdxOut = nullptr;
    double* d_valOut = nullptr;
    int* d_rowPtrIn = nullptr;
    int* d_colIdxIn = nullptr;
    double* d_valIn = nullptr;
    int* d_node_mask = nullptr;
    double* d_constraints = nullptr;
    double* d_sum = nullptr;

    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    int maxThreadsPerBlock = prop.maxThreadsPerBlock;
    int maxBlocks = prop.maxGridSize[0];
    int NODES_PER_BLOCK =128;
    int blockSize = (num_nodes < maxThreadsPerBlock) ? num_nodes : 256;
    int numBlocks = (num_nodes + blockSize - 1) / blockSize;

    if (numBlocks > maxBlocks) {
        NODES_PER_BLOCK = numBlocks / maxBlocks + 1;
        numBlocks = (num_nodes + NODES_PER_BLOCK * blockSize - 1) / (NODES_PER_BLOCK * blockSize);
    }

    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_rowPtrOut, len_rowPtrOut * sizeof(int)));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_colIdxOut, len_colIdxOut * sizeof(int)));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_valOut, len_colIdxOut * sizeof(double)));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_rowPtrIn, len_rowPtrOut * sizeof(int)));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_colIdxIn, len_colIdxOut * sizeof(int)));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_valIn, len_colIdxOut * sizeof(double)));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_constraints, num_nodes * sizeof(double)));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_sum, num_nodes * sizeof(double)));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_node_mask, num_nodes * sizeof(int)));

    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_rowPtrOut, rowPtrOut, len_rowPtrOut * sizeof(int), cudaMemcpyHostToDevice));
    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_colIdxOut, colIdxOut, len_colIdxOut * sizeof(int), cudaMemcpyHostToDevice));
    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_valOut, valOut, len_colIdxOut * sizeof(double), cudaMemcpyHostToDevice));
    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_rowPtrIn, rowPtrIn, len_rowPtrOut * sizeof(int), cudaMemcpyHostToDevice));
    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_colIdxIn, colIdxIn, len_colIdxOut * sizeof(int), cudaMemcpyHostToDevice));
    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_valIn, valIn, len_colIdxOut * sizeof(double), cudaMemcpyHostToDevice));
    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_node_mask, node_mask, num_nodes * sizeof(int), cudaMemcpyHostToDevice));

    

    if(is_directed){
        compute_out_in_sum<<<numBlocks, blockSize>>>(d_rowPtrOut, d_valOut, d_rowPtrIn, d_valIn, num_nodes, d_node_mask, d_sum);
        cudaDeviceSynchronize();

        calculate_constraints<<<numBlocks, blockSize>>>(
            num_nodes,
            NODES_PER_BLOCK,
            d_rowPtrOut,
            d_colIdxOut,
            d_valOut,
            d_rowPtrIn,
            d_colIdxIn,
            d_valIn,
            d_sum, 
            d_node_mask,
            d_constraints
        );
        cudaDeviceSynchronize();
        // constraints.resize(num_nodes);
    }else{
        // calculate_constraints<<<grid_size, block_size>>>(d_V, d_E, d_W, num_nodes, d_node_mask, d_constraint_results);
    }
    EXIT_IF_CUDA_FAILED(cudaMemcpy(constraints, d_constraints, num_nodes * sizeof(double), cudaMemcpyDeviceToHost));
exit:

    cudaFree(d_rowPtrOut);
    cudaFree(d_colIdxOut);
    cudaFree(d_valOut);
    cudaFree(d_rowPtrIn);
    cudaFree(d_colIdxIn);
    cudaFree(d_valIn);
    cudaFree(d_sum);
    cudaFree(d_constraints);
    cudaFree(d_node_mask);
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