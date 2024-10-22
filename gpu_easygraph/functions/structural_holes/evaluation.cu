#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#include "common.h"

namespace gpu_easygraph {

static __device__ double mutual_weight(
    const int* row,
    const int* col,
    const double* W,
    int u,
    int v,
    int num_edges,
    bool is_directed
) {
    double a_uv = 0.0, a_vu = 0.0;
    for (int i = 0; i < num_edges; i++) {
        if (row[i] == u && col[i] == v) {
            a_uv = W[i]; // 找到边 (u, v)，记录其权重
        }
        if (row[i] == v && col[i] == u) {
            a_vu = W[i]; // 找到边 (v, u)，记录其权重
        }
    }
    return a_uv + a_vu;
}

static __device__ double normalized_mutual_weight(
    const int* row, 
    const int* col, 
    const double* W, 
    int num_edges,
    int u,
    int v,
    bool is_directed
    // double* nmw_result
    // bool norm 
) {
    double weight_uv = mutual_weight(row, col, W, u, v, num_edges,is_directed);

    double scale = 0.0;

    for (int i = 0; i < num_edges; i++) {
        if (row[i] == u) {
            int neighbor = col[i];
            double weight_uw = mutual_weight(row, col, W, u, neighbor, num_edges,is_directed);
            scale += weight_uw;
        }
        if(is_directed){
            if (col[i] == u) {
                int neighbor = row[i];
                double weight_wu = mutual_weight(row, col, W, u, neighbor, num_edges,is_directed);
                scale += weight_wu;
            }
        }
    }
    if (scale == 0.0) {
        // *nmw_result = 0.0;
        return 0.0;
    } else {
        // *nmw_result = weight_uv / scale;
        return weight_uv / scale;
    }
}

static __device__ double local_constraint(
    const int* row, 
    const int* col, 
    const double* W,
    int num_edges,
    int u,
    int v,
    bool is_directed
    // double* local_constraint
) {
    double direct = normalized_mutual_weight(row,col,W,num_edges,u,v,is_directed);
    double indirect = 0.0;
    for (int i = 0; i < num_edges; i++) {
        if (row[i] == u) {
            int neighbor = col[i];
            indirect += normalized_mutual_weight(row,col,W,num_edges,u,neighbor,is_directed)*normalized_mutual_weight(row,col,W,num_edges,neighbor,v,is_directed);
        }
        if(is_directed){
            if (col[i] == u) {
                int neighbor = row[i];
                indirect += normalized_mutual_weight(row,col,W,num_edges,u,neighbor,is_directed)*normalized_mutual_weight(row,col,W,num_edges,neighbor,v,is_directed);
            }
        }
    }
    double local_constraint_of_uv = (direct + indirect) * (direct + indirect);
    // printf("Local constraint of (%d, %d): direct = %f, indirect = %f, local_constraint = %f\n",
    //        u, v, direct, indirect, local_constraint_of_uv);
    return local_constraint_of_uv;
    // *local_constraint = constraint_of_v;
}

__global__ void calculate_constraints(
    const int* row, 
    const int* col, 
    const double* W,  
    int num_nodes,
    int num_edges,
    bool is_directed,
    double* constraint_results
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_nodes) return;
    double constraint_of_v = 0.0;
    bool is_nan = true;
    for (int i = 0; i < num_edges; i++) {
        if (row[i] == v) {
            is_nan = false;
            int neighbor = col[i];
            constraint_of_v += local_constraint(row,col,W,num_edges,v,neighbor,is_directed);
        }
        if(is_directed){
            if (col[i] == v) {
                int neighbor = row[i];
                constraint_of_v += local_constraint(row,col,W,num_edges,v,neighbor,is_directed);
            }
        }
    }
    if(is_nan==true){
        constraint_results[v] = NAN;
    }else{
        constraint_results[v] = constraint_of_v;
    }
}

int cuda_constraint(
    _IN_ const int* row,
    _IN_ const int* col,
    _IN_ const double* W,
    _IN_ int num_nodes,
    _IN_ int num_edges,
    _IN_ bool is_directed,
    _OUT_ double* constraint_results
) {
    int* d_row;
    int* d_col;
    double* d_W;
    double* d_constraint_results;

    // 分配CUDA设备内存
    cudaMalloc((void**)&d_row, num_edges * sizeof(int));
    cudaMalloc((void**)&d_col, num_edges * sizeof(int));
    cudaMalloc((void**)&d_W, num_edges * sizeof(double));
    cudaMalloc((void**)&d_constraint_results, num_nodes * sizeof(double));

    // 将数据从主机拷贝到设备
    cudaMemcpy(d_row, row, num_edges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col, col, num_edges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W, W, num_edges * sizeof(double), cudaMemcpyHostToDevice);

    // // 计算归一化权重
    // normalized_mutual_weight<<<1, 1>>>(d_row, d_col, d_W, d_nmw_results);

    // // 对所有节点 v_id 计算约束
    // int threadsPerBlock = 256;
    // int blocksPerGrid = (len_V + threadsPerBlock - 1) / threadsPerBlock;
    calculate_constraints<<<1, num_nodes>>>(d_row, d_col, d_W, num_nodes, num_edges, is_directed, d_constraint_results);

    // // 将结果从设备拷贝回主机
    cudaMemcpy(constraint_results, d_constraint_results, num_nodes * sizeof(double), cudaMemcpyDeviceToHost);

    // 释放设备内存
    cudaFree(d_row);
    cudaFree(d_col);
    cudaFree(d_W);
    cudaFree(d_constraint_results);

    return 0; 
}

} // namespace gpu_easygraph