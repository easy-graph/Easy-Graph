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
    for (int idx = V[u]; idx < V[u + 1]; ++idx) {
        if (E[idx] == v) {
            a_uv = W[idx];
            break;
        }
    }
    for (int idx = V[v]; idx < V[v + 1]; ++idx) {
        if (E[idx] == u) {
            a_vu = W[idx];
            break;
        }
    }
    return a_uv + a_vu;
}

__global__ void normalized_mutual_weight(
    const int* V, 
    const int* E, 
    const double* W, 
    double* nmw_results, 
    int num_nodes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) return;
    int u = idx;
    double scale; 
    for (int i = V[u]; i < V[u + 1]; ++i) {
        int v = E[i]; 
        double weight_uv = mutual_weight(V, E, W, u, v); 
        scale = 0.0; 
        for (int j = V[u]; j < V[u + 1]; ++j) {
            int k = E[j]; 
            scale += mutual_weight(V, E, W, u, k); 
        }

        if (scale == 0) {
            nmw_results[i] = 0.0;
        } else {
            nmw_results[i] = weight_uv / scale; 
        }
    }
}

static __device__ void calculate_constraint_of_v(
    const int* V, 
    const int* E, 
    const double* W, 
    const double* nmw_results, 
    int v_id, 
    double* constraint_result
) {

    double constraint_of_v = 0.0;

    for (int i = V[v_id]; i < V[v_id + 1]; ++i) {
        int v = E[i];
        if(v == v_id) continue;
        double direct = nmw_results[i];  // get p_{uv}

        double indirect_weight = 0.0;

        for (int j = V[v_id]; j < V[v_id + 1]; ++j) {
            int w = E[j];
            double p_uw = nmw_results[j];

            for (int k = V[w]; k < V[w + 1]; ++k) {
                if (E[k] == v) {
                    double p_wv = nmw_results[k];     
                    indirect_weight += p_uw * p_wv;
                }
            }
        }
        double local_constraint_of_uv = (direct + indirect_weight) * (direct + indirect_weight);
        constraint_of_v += local_constraint_of_uv;
    }

    *constraint_result = constraint_of_v;
}

__global__ void calculate_constraints(
    const int* V, 
    const int* E, 
    const double* W, 
    const double* nmw_results, 
    int num_nodes,
    double* constraint_results
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) return;
    calculate_constraint_of_v(V, E, W, nmw_results, idx, &constraint_results[idx]);
}

int cuda_constraint(
    _IN_ const int* V,
    _IN_ const int* E,
    _IN_ const double* W,
    _IN_ int len_V,
    _IN_ int len_E,
    _OUT_ double* constraint_results
) {
    int* d_V;
    int* d_E;
    double* d_W;
    double* d_nmw_results;
    double* d_constraint_results;

    // 分配CUDA设备内存
    cudaMalloc((void**)&d_V, (len_V + 1) * sizeof(int));
    cudaMalloc((void**)&d_E, len_E * sizeof(int));
    cudaMalloc((void**)&d_W, len_E * sizeof(double));
    cudaMalloc((void**)&d_nmw_results, len_E * sizeof(double));
    cudaMalloc((void**)&d_constraint_results, len_V * sizeof(double));

    // 将数据从主机拷贝到设备
    cudaMemcpy(d_V, V, (len_V + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_E, E, len_E * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W, W, len_E * sizeof(double), cudaMemcpyHostToDevice);

    // 计算归一化权重
    normalized_mutual_weight<<<1, len_V + 1>>>(d_V, d_E, d_W, d_nmw_results, len_V);

    // 对所有节点 v_id 计算约束
    int threadsPerBlock = 256;
    int blocksPerGrid = (len_V + threadsPerBlock - 1) / threadsPerBlock;
    calculate_constraints<<<blocksPerGrid, threadsPerBlock>>>(d_V, d_E, d_W, d_nmw_results, len_V, d_constraint_results);

    // 将结果从设备拷贝回主机
    cudaMemcpy(constraint_results, d_constraint_results, len_V * sizeof(double), cudaMemcpyDeviceToHost);

    // 释放设备内存
    cudaFree(d_V);
    cudaFree(d_E);
    cudaFree(d_W);
    cudaFree(d_nmw_results);
    cudaFree(d_constraint_results);

    return 0; 
}

} // namespace gpu_easygraph