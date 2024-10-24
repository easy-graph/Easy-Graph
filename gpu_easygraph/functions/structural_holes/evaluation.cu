#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#include "common.h"

namespace gpu_easygraph {

// static __device__ double mutual_weight(
//     const int* row,
//     const int* col,
//     const double* W,
//     int u,
//     int v,
//     int num_edges,
//     bool is_directed
// ) {
//     double a_uv = 0.0, a_vu = 0.0;
//     for (int i = 0; i < num_edges; i++) {
//         if (row[i] == u && col[i] == v) {
//             a_uv = W[i]; // 找到边 (u, v)，记录其权重
//         }
//         if (row[i] == v && col[i] == u) {
//             a_vu = W[i]; // 找到边 (v, u)，记录其权重
//         }
//     }
//     return a_uv + a_vu;
// }

static __device__ double mutual_weight(
    const int* V,
    const int* E,
    const double* W,
    int u,
    int v,
    int num_edges,
    bool is_directed
) {
    double a_uv = 0.0, a_vu = 0.0;

    // 查找 u->v 的权重
    for (int i = V[u]; i < V[u+1]; i++) {
        if (E[i] == v) {
            a_uv = W[i];
            break;
        }
    }
    if(is_directed){
    // 查找 v->u 的权重（如果是无向图）
        for (int i = V[v]; i < V[v+1]; i++) {
            if (E[i] == u) {
                a_vu = W[i];
                break;
            }
        }
    }
    return a_uv + a_vu;
}

static __device__ double normalized_mutual_weight(
    const int* V,
    const int* E,
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
    double weight_uv = mutual_weight(V, E, W, u, v, num_edges,is_directed);

    double scale = 0.0;
    // 遍历节点 u 的所有出邻居（使用 CSR）
    for (int i = V[u]; i < V[u+1]; i++) {
        int neighbor = E[i];  // 找到 u 的一个出邻居
        double weight_uw = mutual_weight(V, E, W, u, neighbor, num_edges,is_directed);
        scale += weight_uw;   // 累积权重
    }

    // 如果是有向图，还需要处理 u 的入邻居（使用 COO）
    if (is_directed) {
        for (int i = 0; i < num_edges; i++) {
            if (col[i] == u) {  // 找到 u 的入邻居
                int neighbor = row[i];  // 该入邻居是 row[i]
                double weight_wu = mutual_weight(V, E, W, u, neighbor, num_edges,is_directed);
                scale += weight_wu;  // 累积入邻居的权重
            }
        }
    }
    // for (int i = 0; i < num_edges; i++) {
    //     if (row[i] == u) {
    //         int neighbor = col[i];
    //         double weight_uw = mutual_weight(V, E, W, u, neighbor, num_edges,is_directed);
    //         scale += weight_uw;
    //     }
    //     if(is_directed){
    //         if (col[i] == u) {
    //             int neighbor = row[i];
    //             double weight_wu = mutual_weight(V, E, W, u, neighbor, num_edges,is_directed);
    //             scale += weight_wu;
    //         }
    //     }
    // }
    const double epsilon = 1e-16;
    // if (scale == 0.0) {
    if (scale < epsilon) {
        // *nmw_result = 0.0;
        return 0.0;
    } else {
        // *nmw_result = weight_uv / scale;
        return weight_uv / scale;
    }
}

static __device__ double local_constraint(
    const int* V,
    const int* E,
    const int* row, 
    const int* col, 
    const double* W,
    int num_edges,
    int u,
    int v,
    bool is_directed
    // double* local_constraint
) {
    double direct = normalized_mutual_weight(V,E,row,col,W,num_edges,u,v,is_directed);
    double indirect = 0.0;
    for (int i = V[u]; i < V[u+1]; i++) {
        int neighbor = E[i];
        // 计算 u -> neighbor 和 neighbor -> v 的归一化权重
        double norm_uw = normalized_mutual_weight(V, E, row, col, W, num_edges, u, neighbor, is_directed);
        double norm_wv = normalized_mutual_weight(V, E, row, col, W, num_edges, neighbor, v, is_directed);
        indirect += norm_uw * norm_wv;
    }

    // 如果是有向图，还需要遍历入邻居，使用 COO
    if (is_directed) {
        for (int i = 0; i < num_edges; i++) {
            if (col[i] == u) {  // 查找 u 的入邻居
                int neighbor = row[i];
                double norm_uw = normalized_mutual_weight(V, E, row, col, W, num_edges, u, neighbor, is_directed);
                double norm_wv = normalized_mutual_weight(V, E, row, col, W, num_edges, neighbor, v, is_directed);
                indirect += norm_uw * norm_wv;
            }
        }
    }
    // for (int i = 0; i < num_edges; i++) {
    //     if (row[i] == u) {
    //         int neighbor = col[i];
    //         indirect += normalized_mutual_weight(V,E,row,col,W,num_edges,u,neighbor,is_directed)*normalized_mutual_weight(V,E,row,col,W,num_edges,neighbor,v,is_directed);
    //     }
    //     if(is_directed){
    //         if (col[i] == u) {
    //             int neighbor = row[i];
    //             indirect += normalized_mutual_weight(V,E,row,col,W,num_edges,u,neighbor,is_directed)*normalized_mutual_weight(V,E,row,col,W,num_edges,neighbor,v,is_directed);
    //         }
    //     }
    // }
    // 同步线程，确保所有线程完成权重计算
    // __syncthreads();
    double local_constraint_of_uv = (direct + indirect) * (direct + indirect);
    // printf("Local constraint of (%d, %d): direct = %f, indirect = %f, local_constraint = %f\n",
    //        u, v, direct, indirect, local_constraint_of_uv);
    return local_constraint_of_uv;
    // *local_constraint = constraint_of_v;
}

__global__ void calculate_constraints(
    const int* V,
    const int* E,
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
    for (int i = V[v]; i < V[v+1]; i++) {
        is_nan = false;
        int neighbor = E[i];  // v 的出邻居

        // 计算 local constraint，传递 CSR 和 COO 结构
        constraint_of_v += local_constraint(V, E, row, col, W, num_edges, v, neighbor, is_directed);
    }

    // 如果是有向图，还需要处理 v 的入邻居（通过 COO 遍历）
    if (is_directed) {
        for (int i = 0; i < num_edges; i++) {
            if (col[i] == v) {  // 查找 v 的入邻居
                // is_nan = false;
                int neighbor = row[i];  // 入邻居
                constraint_of_v += local_constraint(V, E, row, col, W, num_edges, v, neighbor, is_directed);
            }
        }
    }
    // for (int i = 0; i < num_edges; i++) {
    //     if (row[i] == v) {
    //         is_nan = false;
    //         int neighbor = col[i];
    //         constraint_of_v += local_constraint(V,E,row,col,W,num_edges,v,neighbor,is_directed);
    //     }
    //     if(is_directed){
    //         if (col[i] == v) {
    //             int neighbor = row[i];
    //             constraint_of_v += local_constraint(V,E,row,col,W,num_edges,v,neighbor,is_directed);
    //         }
    //     }
    // }
    // 在线程块内部同步，确保所有线程完成计算再写入结果
    // __syncthreads();
    if(is_nan){
        constraint_results[v] = NAN;
    }else{
        constraint_results[v] = constraint_of_v;
    }
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
    int* d_V;
    int* d_E;
    int* d_row;
    int* d_col;
    double* d_W;
    double* d_constraint_results;
    int threadsPerBlock = 512;
    int blocksPerGrid = (num_nodes + threadsPerBlock - 1) / threadsPerBlock;

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

    // // 计算归一化权重
    // normalized_mutual_weight<<<1, 1>>>(d_row, d_col, d_W, d_nmw_results);

    // // 对所有节点 v_id 计算约束
    
    calculate_constraints<<<blocksPerGrid, threadsPerBlock>>>(d_V, d_E, d_row, d_col, d_W, num_nodes, num_edges, is_directed, d_constraint_results);

    // // 将结果从设备拷贝回主机
    EXIT_IF_CUDA_FAILED(cudaMemcpy(constraint_results, d_constraint_results, num_nodes * sizeof(double), cudaMemcpyDeviceToHost));
exit:
    // 释放设备内存
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