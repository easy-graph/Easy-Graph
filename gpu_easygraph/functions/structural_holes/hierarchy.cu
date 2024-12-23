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

__global__ void calculate_hierarchy(
    const int* V, 
    const int* E, 
    const double* W,
    int num_nodes,
    const int* node_mask,
    double* hierarchy_results
) {
    int start_node = blockIdx.x * NODES_PER_BLOCK;
    int end_node = min(start_node + NODES_PER_BLOCK, num_nodes);

    extern __shared__ double shared_mem[];
    double* shared_c = shared_mem;  
    double* shared_C = &shared_mem[blockDim.x]; 

    for (int v = start_node; v < end_node; ++v) {
        if (!node_mask[v]) continue;

        int n = V[v + 1] - V[v]; 
        if (n <= 1) {
            hierarchy_results[v] = 0.0;
            continue;
        }
        if (threadIdx.x == 0) shared_C[0] = 0.0;
        __syncthreads();

        double local_C = 0.0;
        
        for (int i = V[v] + threadIdx.x; i < V[v + 1]; i += blockDim.x) {
            int w = E[i];
            double constraint = local_constraint(V, E, W, v, w); 
            shared_c[threadIdx.x] = constraint;  
            local_C += constraint; 
        }

        atomicAdd(&shared_C[0], local_C);
        __syncthreads();

        if (threadIdx.x == 0) { 
            double C = shared_C[0]; 
            double hierarchy_sum = 0.0;
            for (int i = 0; i < n; i++) {
                double normalized_c = shared_c[i] / C;
                hierarchy_sum += normalized_c * n * logf(normalized_c * n) / (n * logf(n));
            }
            hierarchy_results[v] = hierarchy_sum; 
        }

        __syncthreads();  
    }
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
    
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, calculate_hierarchy, 0, 0);
    int grid_size = (num_nodes + block_size - 1) / block_size;
    
    int* d_V;
    int* d_E;
    int* d_row;
    int* d_col;
    double* d_W;
    int* d_node_mask;
    double* d_hierarchy_results;

    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_V, (num_nodes+1) * sizeof(int)));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_E, num_edges * sizeof(int)));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_row, num_edges * sizeof(int)));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_col, num_edges * sizeof(int)));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_W, num_edges * sizeof(double)));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_node_mask, num_nodes * sizeof(int)));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_hierarchy_results, num_nodes * sizeof(double)));

    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_V, V, (num_nodes+1) * sizeof(int), cudaMemcpyHostToDevice));
    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_E, E, num_edges * sizeof(int), cudaMemcpyHostToDevice));
    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_row, row, num_edges * sizeof(int), cudaMemcpyHostToDevice));
    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_col, col, num_edges * sizeof(int), cudaMemcpyHostToDevice));
    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_node_mask, node_mask, num_nodes * sizeof(int), cudaMemcpyHostToDevice));
    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_W, W, num_edges * sizeof(double), cudaMemcpyHostToDevice));

    if(is_directed){
        directed_calculate_hierarchy<<<grid_size, block_size>>>(d_V, d_E, d_row, d_col, d_W, num_nodes, num_edges, d_node_mask, d_hierarchy_results);
    }else{
        int block_size = 256;
        int grid_size = (num_nodes + NODES_PER_BLOCK - 1) / NODES_PER_BLOCK; 
        int shared_memory_size = 2 * sizeof(double) * block_size; 
        calculate_hierarchy<<<grid_size, block_size, shared_memory_size>>>(d_V, d_E, d_W, num_nodes, d_node_mask, d_hierarchy_results);
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
}