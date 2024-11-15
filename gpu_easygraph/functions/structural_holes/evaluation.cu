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

static __device__ double local_constraint(
    const int* V,
    const int* E,
    const double* W,
    int u,
    int v
) {
    double direct = normalized_mutual_weight(V,E,W,u,v,SUM);
    double indirect = 0.0;
    for (int i = V[u]; i < V[u+1]; i++) {
        int neighbor = E[i];
        double norm_uw = normalized_mutual_weight(V, E, W, u, neighbor,SUM);
        double norm_wv = normalized_mutual_weight(V, E, W, neighbor, v,SUM);
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
    const int* node_mask,
    double* constraint_results
) {
    int start_node = blockIdx.x * NODES_PER_BLOCK;  // 当前块处理的起始节点
    int end_node = min(start_node + NODES_PER_BLOCK, num_nodes);  // 结束节点，确保不越界

    for (int v = start_node; v < end_node; ++v) {
        if (!node_mask[v]) continue;

        double constraint_of_v = 0.0;
        bool is_nan = true;

        // 使用所有线程并行处理邻居
        __shared__ double shared_constraint[256];  // 假设 blockDim.x <= 256
        double local_sum = 0.0;

        for (int i = V[v] + threadIdx.x; i < V[v + 1]; i += blockDim.x) {
            is_nan = false;
            int neighbor = E[i];
            local_sum += local_constraint(V, E, W, v, neighbor);
        }

        // 归约邻居结果到一个值
        shared_constraint[threadIdx.x] = local_sum;
        __syncthreads();

        for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
            if (threadIdx.x < offset) {
                shared_constraint[threadIdx.x] += shared_constraint[threadIdx.x + offset];
            }
            __syncthreads();
        }

        // 保存最终结果
        if (threadIdx.x == 0) {
            constraint_results[v] = (is_nan) ? NAN : shared_constraint[0];
        }
    }
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
    double direct = directed_normalized_mutual_weight(V,E,row,col,W,num_edges,u,v,SUM);
    double indirect = 0.0;
    for (int i = V[u]; i < V[u+1]; i++) {
        int neighbor = E[i];
        double norm_uw = directed_normalized_mutual_weight(V, E, row, col, W, num_edges, u, neighbor,SUM);
        double norm_wv = directed_normalized_mutual_weight(V, E, row, col, W, num_edges, neighbor, v,SUM);
        indirect += norm_uw * norm_wv;
    }

    for (int i = 0; i < num_edges; i++) {
        if (col[i] == u) {
            int neighbor = row[i];
            double norm_uw = directed_normalized_mutual_weight(V, E, row, col, W, num_edges, u, neighbor,SUM);
            double norm_wv = directed_normalized_mutual_weight(V, E, row, col, W, num_edges, neighbor, v,SUM);
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
    int* node_mask,
    double* constraint_results
) {
    int start_node = blockIdx.x * NODES_PER_BLOCK;  // 当前块处理的起始节点
    int end_node = min(start_node + NODES_PER_BLOCK, num_nodes);  // 结束节点，确保不越界

    for (int v = start_node; v < end_node; ++v) {
        if (!node_mask[v]) continue;

        double constraint_of_v = 0.0;
        bool is_nan = true;

        // 使用所有线程并行处理邻居
        __shared__ double shared_constraint[256];  // 假设 blockDim.x <= 256
        double local_sum = 0.0;

        // 计算出邻居约束（out-neighbors）
        for (int i = V[v] + threadIdx.x; i < V[v + 1]; i += blockDim.x) {
            is_nan = false;
            int neighbor = E[i];
            local_sum += directed_local_constraint(V, E, row, col, W, num_edges, v, neighbor);
        }

        // 计算入邻居约束（in-neighbors）
        for (int i = threadIdx.x; i < num_edges; i += blockDim.x) {
            if (col[i] == v) {
                // is_nan = false;
                int neighbor = row[i];
                local_sum += directed_local_constraint(V, E, row, col, W, num_edges, v, neighbor);
            }
        }

        // 将所有线程的局部结果写入共享内存
        shared_constraint[threadIdx.x] = local_sum;
        __syncthreads();

        // 归约邻居结果到一个值
        for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
            if (threadIdx.x < offset) {
                shared_constraint[threadIdx.x] += shared_constraint[threadIdx.x + offset];
            }
            __syncthreads();
        }

        // 保存最终结果
        if (threadIdx.x == 0) {
            constraint_results[v] = (is_nan) ? NAN : shared_constraint[0];
        }
    }
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

// __global__ void calculate_effective_size(
//     const int* V,
//     const int* E,
//     const double* W,
//     const int num_nodes,
//     const int* node_mask,
//     double* effective_size_results
// ) {
//     int u = blockIdx.x * blockDim.x + threadIdx.x;
//     if (u >= num_nodes || !node_mask[u]) return;
//     double redundancy_sum = 0.0;
//     bool is_nan = true;

//     // 遍历 v 的所有邻居
//     for (int i = V[u]; i < V[u + 1]; i++) {
//         int v = E[i];
//         if (v == u) continue; // 排除自连接的情况
//         is_nan = false;
//         redundancy_sum += redundancy(V,E,W,num_nodes,u,v);
//     }
//     effective_size_results[u] = is_nan ? NAN : redundancy_sum;
// }

__global__ void calculate_effective_size(
    const int* V,
    const int* E,
    const double* W,
    const int num_nodes,
    const int* node_mask,
    double* effective_size_results
) {
    int start_node = blockIdx.x * NODES_PER_BLOCK;  // 当前块处理的起始节点
    int end_node = min(start_node + NODES_PER_BLOCK, num_nodes);  // 结束节点，确保不越界

    for (int u = start_node; u < end_node; ++u) {
        if (!node_mask[u]) continue;

        bool is_nan = true;  // 用于标记是否存在有效邻居
        extern __shared__ double shared_redundancy[];  // 假设 blockDim.x <= 256
        double local_redundancy_sum = 0.0;

        // 并行处理 u 的所有邻居
        for (int i = V[u] + threadIdx.x; i < V[u + 1]; i += blockDim.x) {
            int v = E[i];
            if (v == u) continue;  // 排除自连接的情况
            is_nan = false;  // 标记 u 有有效邻居
            local_redundancy_sum += redundancy(V, E, W, num_nodes, u, v);
        }

        // 将线程的局部结果写入共享内存
        shared_redundancy[threadIdx.x] = local_redundancy_sum;
        __syncthreads();

        // 归约邻居的冗余值到一个总值
        for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
            if (threadIdx.x < offset) {
                shared_redundancy[threadIdx.x] += shared_redundancy[threadIdx.x + offset];
            }
            __syncthreads();
        }

        // 保存最终结果
        if (threadIdx.x == 0) {
            effective_size_results[u] = is_nan ? NAN : shared_redundancy[0];
        }
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

    // 遍历 u 的所有邻居
    for (int i = V[u]; i < V[u + 1]; i++) {
        int v = E[i];
        if (v == u) continue; // 排除自连接的情况
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

__global__ void calculate_hierarchy(
    const int* V, 
    const int* E, 
    const double* W,
    int num_nodes,
    const int* node_mask,
    double* hierarchy_results
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_nodes || !node_mask[v]) return;
    int n = V[v + 1] - V[v];  
    double *c = new double[n]; 
    double C = 0.0;
    double hierarchy_sum = 0.0;
    int neighbor = 0;
    for (int i = V[v]; i < V[v + 1]; i++) {
        int w = E[i];
        c[neighbor] = local_constraint(V, E, W, v, w);
        C += c[neighbor++];
    }
    __syncthreads();
    if (n > 1) {
        for (int i = 0; i < neighbor; i++) {
            hierarchy_sum += (c[i] / C) * n * logf((c[i] / C) * n) / (n * logf(n));
        }
        hierarchy_results[v] = hierarchy_sum;
    }else{
        hierarchy_results[v] = 0;
    }
    delete[] c;
}

static __device__ double atomicAdd (
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

__global__ void directed_calculate_hierarchy(
    const int* V,
    const int* E,
    const int* row,
    const int* col,
    const double* W, 
    const int num_nodes,
    const int num_edges,
    const int* node_mask,
    double* hierarchy_results
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_nodes || !node_mask[v]) return;
    int in_neighbor = V[v + 1] - V[v];  
    int out_neighbor = 0;
    double C = 0.0;
    double hierarchy_sum = 0.0;
    int neighbor = 0;
    for (int i = 0; i < num_edges; i++) {
        if (col[i] == v) {
            out_neighbor++;
        }
    }
    double *c = new double[in_neighbor+out_neighbor]; 
    for (int i = V[v]; i < V[v + 1]; i++) {
        int w = E[i];
        c[neighbor] = directed_local_constraint(V, E, row, col, W, num_edges, v, w);
        C += c[neighbor];
        neighbor++;
    }
    for (int i = 0; i < num_edges; i++) {
        if (col[i] == v) {
            int w = row[i];
            c[neighbor] = directed_local_constraint(V, E, row, col, W, num_edges, v, w);
            C += c[neighbor];
            neighbor++;
        }
    }
    __syncthreads();
    if (neighbor > 1) {
        for (int i = 0; i < neighbor; i++) {
            hierarchy_sum += (c[i] / C) * neighbor * logf((c[i] / C) * neighbor) / (neighbor * logf(neighbor));
        }
        hierarchy_results[v] = hierarchy_sum;
    }else{
        hierarchy_results[v] = 0;
    }
    delete[] c;
}



int cuda_hierarchy(
    _IN_ const int* V,
    _IN_ const int* E,
    _IN_ const int* row,
    _IN_ const int* col,
    _IN_ const double* W,
    _IN_ int num_nodes,
    _IN_ int num_edges,
    _IN_ bool is_directed,
    _IN_ int* node_mask,
    _OUT_ double* hierarchy_results
){
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
    int* d_node_mask;
    double* d_hierarchy_results;

    // 分配CUDA设备内存
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_V, (num_nodes+1) * sizeof(int)));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_E, num_edges * sizeof(int)));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_row, num_edges * sizeof(int)));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_col, num_edges * sizeof(int)));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_W, num_edges * sizeof(double)));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_node_mask, num_nodes * sizeof(int)));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_hierarchy_results, num_nodes * sizeof(double)));

    // 将数据从主机拷贝到设备
    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_V, V, (num_nodes+1) * sizeof(int), cudaMemcpyHostToDevice));
    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_E, E, num_edges * sizeof(int), cudaMemcpyHostToDevice));
    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_row, row, num_edges * sizeof(int), cudaMemcpyHostToDevice));
    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_col, col, num_edges * sizeof(int), cudaMemcpyHostToDevice));
    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_node_mask, node_mask, num_nodes * sizeof(int), cudaMemcpyHostToDevice));
    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_W, W, num_edges * sizeof(double), cudaMemcpyHostToDevice));


    // 调用 CUDA 内核
    if(is_directed){
        directed_calculate_hierarchy<<<grid_size, block_size>>>(d_V, d_E, d_row, d_col, d_W, num_nodes, num_edges, d_node_mask, d_hierarchy_results);
    }else{
        // int block_size = 256;  // 每个块的线程数，可以根据设备进行调整
        // int grid_size = (num_nodes + NODES_PER_BLOCK - 1) / NODES_PER_BLOCK;  // 调整为每个块处理NODES_PER_BLOCK个节点
        // int shared_memory_size = block_size * sizeof(double); 
        calculate_hierarchy<<<grid_size, block_size>>>(d_V, d_E, d_W, num_nodes, d_node_mask, d_hierarchy_results);
    }

    EXIT_IF_CUDA_FAILED(cudaMemcpy(hierarchy_results, d_hierarchy_results, num_nodes * sizeof(double), cudaMemcpyDeviceToHost));
exit:

    cudaFree(d_V);
    cudaFree(d_E);
    cudaFree(d_row);
    cudaFree(d_col);
    cudaFree(d_W);
    cudaFree(d_node_mask);
    cudaFree(d_hierarchy_results);
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

    
    // cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, 
    // is_directed ? directed_calculate_effective_size : undirected_calculate_effective_size, 0, 0);
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, calculate_effective_size, 0, 0);
    int grid_size = (num_nodes + block_size - 1) / block_size;

    int* d_V;
    int* d_E;
    int* d_row;
    int* d_col;
    double* d_W;
    int* d_node_mask;
    double* d_effective_size_results;

    // 分配 CUDA 设备内存
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_V, (num_nodes+1) * sizeof(int)));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_E, num_edges * sizeof(int)));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_row, num_edges * sizeof(int)));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_col, num_edges * sizeof(int)));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_W, num_edges * sizeof(double)));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_node_mask, num_nodes * sizeof(int)));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_effective_size_results, num_nodes * sizeof(double)));

    // 将数据从主机拷贝到设备
    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_V, V, (num_nodes+1) * sizeof(int), cudaMemcpyHostToDevice));
    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_E, E, num_edges * sizeof(int), cudaMemcpyHostToDevice));
    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_row, row, num_edges * sizeof(int), cudaMemcpyHostToDevice));
    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_col, col, num_edges * sizeof(int), cudaMemcpyHostToDevice));
    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_node_mask, node_mask, num_nodes * sizeof(int), cudaMemcpyHostToDevice));
    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_W, W, num_edges * sizeof(double), cudaMemcpyHostToDevice));

    if(is_directed){
        directed_calculate_effective_size<<<grid_size, block_size>>>(d_V, d_E, d_row, d_col, d_W, num_nodes, num_edges, d_node_mask, d_effective_size_results);
    }else{
        int block_size = 256;  // 每个块的线程数，可以根据设备进行调整
        int grid_size = (num_nodes + NODES_PER_BLOCK - 1) / NODES_PER_BLOCK;  // 调整为每个块处理NODES_PER_BLOCK个节点
        int shared_memory_size = block_size * sizeof(double); 
        calculate_effective_size<<<grid_size, block_size,shared_memory_size>>>(d_V, d_E, d_W, num_nodes, d_node_mask, d_effective_size_results);
        
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

int cuda_constraint(
    _IN_ const int* V,
    _IN_ const int* E,
    _IN_ const int* row,
    _IN_ const int* col,
    _IN_ const double* W,
    _IN_ int num_nodes,
    _IN_ int num_edges,
    _IN_ bool is_directed,
    _IN_ int* node_mask,
    _OUT_ double* constraint_results
) {
    int cuda_ret = cudaSuccess;
    int EG_ret = EG_GPU_SUCC;
    
    

    // calculate_constraints<<<grid_size, block_size>>>(d_V, d_E, d_W, num_nodes, d_node_mask, d_constraint_results);
    
    int* d_V;
    int* d_E;
    int* d_row;
    int* d_col;
    double* d_W;
    int* d_node_mask;
    double* d_constraint_results;
    int block_size = 256;  // 每个块的线程数，可以根据设备进行调整
    int grid_size = (num_nodes + NODES_PER_BLOCK - 1) / NODES_PER_BLOCK;  // 调整为每个块处理NODES_PER_BLOCK个节点

    // 分配CUDA设备内存
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_V, (num_nodes+1) * sizeof(int)));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_E, num_edges * sizeof(int)));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_row, num_edges * sizeof(int)));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_col, num_edges * sizeof(int)));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_W, num_edges * sizeof(double)));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_node_mask, num_nodes * sizeof(int)));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_constraint_results, num_nodes * sizeof(double)));

    // 将数据从主机拷贝到设备
    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_V, V, (num_nodes+1) * sizeof(int), cudaMemcpyHostToDevice));
    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_E, E, num_edges * sizeof(int), cudaMemcpyHostToDevice));
    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_row, row, num_edges * sizeof(int), cudaMemcpyHostToDevice));
    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_col, col, num_edges * sizeof(int), cudaMemcpyHostToDevice));
    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_node_mask, node_mask, num_nodes * sizeof(int), cudaMemcpyHostToDevice));
    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_W, W, num_edges * sizeof(double), cudaMemcpyHostToDevice));

    
    // 调用 CUDA 内核
    if(is_directed){
        // int min_grid_size = 0;
        // int block_size = 0;
        // cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, calculate_constraints, 0, 0);
        // int grid_size = (num_nodes + block_size - 1) / block_size;
        directed_calculate_constraints<<<grid_size, block_size>>>(d_V, d_E, d_row, d_col, d_W, num_nodes, num_edges, d_node_mask, d_constraint_results);
    }else{
        
        calculate_constraints<<<grid_size, block_size>>>(d_V, d_E, d_W, num_nodes, d_node_mask, d_constraint_results);
    }

    EXIT_IF_CUDA_FAILED(cudaMemcpy(constraint_results, d_constraint_results, num_nodes * sizeof(double), cudaMemcpyDeviceToHost));
exit:

    cudaFree(d_V);
    cudaFree(d_E);
    cudaFree(d_row);
    cudaFree(d_col);
    cudaFree(d_W);
    cudaFree(d_node_mask);
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