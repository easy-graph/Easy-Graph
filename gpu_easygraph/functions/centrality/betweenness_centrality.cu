#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#include "common.h"

static __device__ double atomicAddDouble (
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



static __device__ double atomicMinDouble (
    _OUT_ double *address, 
    _IN_ double val
)
{
	unsigned long long ret = __double_as_longlong(*address);
	while (val < __longlong_as_double(ret))
	{
		unsigned long long old = ret;
		if ((ret = atomicCAS((unsigned long long *)address, old, __double_as_longlong(val))) == old)
			break;
	}
	return __longlong_as_double(ret);
}



static __global__ void d_calc_min_edge (
    _IN_ int* d_V,
    _IN_ int* d_E,
    _IN_ double* d_W,
    _IN_ int len_V,
    _IN_ int len_E,
    _OUT_ double* d_min_edge
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < len_V) {
		double curr_min = EG_DOUBLE_INF;
        int edge_start = d_V[tid];
        int edge_end = tid + 1 < len_V ? d_V[tid + 1] : len_E;
		for(int i = edge_start; i < edge_end; ++i) {
            curr_min = min(curr_min, d_W[i]);
		}
		d_min_edge[tid] = curr_min;
	}
}



static __global__ void d_dijkstra_bc (
    _IN_ int* d_V,
    _IN_ int* d_E,
    _IN_ double* d_W,
    _IN_ double* d_min_edge,
    _IN_ int* d_sources,
    _BUFFER_ double* d_dist_2D,
    _BUFFER_ double* d_sigma_2D,
    _BUFFER_ double* d_delta_2D,
    _BUFFER_ int* d_U_2D,
    _BUFFER_ int* d_F_2D,
    _BUFFER_ int* d_st_2D,
    _BUFFER_ int* d_st_idx_2D,
    _IN_ int len_V,
    _IN_ int len_E,
    _IN_ int len_sources,
    _IN_ int warp_size,
    _IN_ int endpoints,
    _OUT_ double* d_BC
)
{
    for (int s_idx = blockIdx.x; s_idx < len_sources; s_idx += gridDim.x) {
        int s = d_sources[s_idx];

        double* d_dist = d_dist_2D + blockIdx.x * len_V;
        double* d_sigma = d_sigma_2D + blockIdx.x * len_V;
        double* d_delta = d_delta_2D + blockIdx.x * len_V;

        int* d_U = d_U_2D + blockIdx.x * len_V;
        int* d_F = d_F_2D + blockIdx.x * len_V;
        int* d_st = d_st_2D + blockIdx.x * len_V;
        int* d_st_idx = d_st_idx_2D + blockIdx.x * (len_V + 2);

        __shared__ int len_F;
        __shared__ int len_st;
        __shared__ int len_st_idx;
        __shared__ double delta;

        for (int i = threadIdx.x; i < len_V; i += blockDim.x) {
            d_dist[i] = EG_DOUBLE_INF;
            d_sigma[i] = 0;
            d_delta[i] = 0;

            d_U[i] = 1;
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            d_dist[s] = 0;
            d_sigma[s] = 1;

            d_U[s] = 0;
            d_F[0] = s;
            len_F = 1;
            d_st[0] = s;
            len_st = 1;
            d_st_idx[0] = 0;
            d_st_idx[1] = 1;
            len_st_idx = 2;

            delta = 0.0;
        }
        __syncthreads();

        while (delta < EG_DOUBLE_INF) {
            for (int j = threadIdx.x; j < len_F * warp_size; j += blockDim.x) {
                int f = d_F[j / warp_size];
                int edge_start = d_V[f];
                int edge_end = f + 1 < len_V ? d_V[f + 1] : len_E;
                double dist = d_dist[f];
                for (int e = j % warp_size; e < edge_end - edge_start; e += warp_size) {
                    int adj = d_E[e + edge_start];
                    double relax_w = dist + d_W[e + edge_start];
                    atomicMinDouble(d_dist + adj, relax_w);
                }
                __threadfence_block();
            }
            __syncthreads();

            if (threadIdx.x == 0) {
                delta = EG_DOUBLE_INF;
            }
            __syncthreads();

            for (int i = threadIdx.x; i < len_V; i += blockDim.x) {
                double dist_i = d_dist[i];
                if (d_U[i] == 1 && dist_i < EG_DOUBLE_INF) {
                    atomicMinDouble(&delta, dist_i + d_min_edge[i]);
                }
            }
            __syncthreads();

            if (threadIdx.x == 0) {
				len_F = 0;
			}
			__syncthreads();

            for (int i = threadIdx.x; i < len_V; i += blockDim.x) {
                double dist_i = d_dist[i];
                if (d_U[i] && dist_i < delta && dist_i < EG_DOUBLE_INF) {
                    d_U[i] = 0;
                    int f_idx = atomicAdd(&len_F, 1);
                    d_F[f_idx] = i;
                }
            }
            __syncthreads();

            for (int i = threadIdx.x; i < len_F; i += blockDim.x) {
                int st_idx = atomicAdd(&len_st, 1);
                d_st[st_idx] = d_F[i];
            }
            __syncthreads();
            
            if (threadIdx.x == 0) {
                d_st_idx[len_st_idx] = d_st_idx[len_st_idx - 1] + len_F;
                ++len_st_idx;
            }
            __syncthreads();
        }
        // calculate single source shortest path END

        // calculate sigma START
        for (int curr_lvl = 0; curr_lvl + 1 < len_st_idx; ++curr_lvl) {
            int lvl_start = d_st_idx[curr_lvl];
            int lvl_end = d_st_idx[curr_lvl + 1];
            for (int j = threadIdx.x; j < (lvl_end - lvl_start) * warp_size; j += blockDim.x) {
                int v = d_st[lvl_start + j / warp_size];
                double dist_v = d_dist[v];
                int edge_start = d_V[v];
                int edge_end = v + 1 < len_V ? d_V[v + 1] : len_E;
                for (int e = j % warp_size; e < edge_end - edge_start; e += warp_size) {
                    int adj = d_E[e + edge_start];
                    if (dist_v + d_W[e + edge_start] == d_dist[adj]) {
                        atomicAddDouble(d_sigma + adj, d_sigma[v]);
                    }
                }
                __threadfence_block();
            }
            __syncthreads();
        }
        // calculate sigma END

        __shared__ int depth, st_start, st_end;
        if (threadIdx.x == 0) {
            depth = len_st_idx - 1;
        }
        __syncthreads();

        if (threadIdx.x == 0 && endpoints) {
            atomicAddDouble(d_BC + s, d_st_idx[depth] - 1);
        }
        __syncthreads();

        while (depth > 0) {
            if (threadIdx.x == 0) {
                st_start = d_st_idx[depth - 1];
                st_end = d_st_idx[depth];
            }
            __syncthreads();

            for (int j = threadIdx.x; j < (st_end - st_start) * warp_size; j += blockDim.x) {
                int pred = d_st[st_start + j / warp_size];
                int edge_start = d_V[pred];
                int edge_end = pred + 1 < len_V ? d_V[pred + 1] : len_E;
                double pred_sigma = d_sigma[pred];
                double pred_dist = d_dist[pred];

                for (int e = j % warp_size; e < edge_end - edge_start; e += warp_size) {
                    int succ = d_E[e + edge_start];
                    double weight = d_W[e + edge_start];
                    double succ_dist = d_dist[succ];
                    if (succ_dist == pred_dist + weight) {
                        atomicAddDouble(d_delta + pred, 
                                pred_sigma / d_sigma[succ] * (1 + d_delta[succ]));
                    }
                }
                __threadfence_block();
                
                if (j % warp_size == 0 && s != pred) {
                    atomicAddDouble(d_BC + pred, d_delta[pred] + endpoints);
                }
            }
            __syncthreads();


            if (threadIdx.x == 0) {
                --depth;
            }
            __syncthreads();
        }
    }
}



static __global__ void d_rescale(
    _IN_ int len_V,
    _IN_ double scale,
    _OUT_ double* d_BC
)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < len_V) {
        d_BC[tid] *= scale;
    }
}



static double calc_scale(
    _IN_ int len_V,
    _IN_ int is_directed,
    _IN_ int normalized,
    _IN_ int endpoints
)
{
    double scale = 1.0;
    if (normalized) {
        if (endpoints) {
            if (len_V < 2) {
                scale = 1.0;
            } else {
                scale = 1.0 / (double(len_V) * (len_V - 1));
            }
        } else if (len_V <= 2) {
            scale = 1.0;
        } else {
            scale = 1.0 / ((double(len_V) - 1) * (len_V - 2));
        }
    } else {
        if (!is_directed) {
            scale = 0.5;
        } else {
            scale = 1.0;
        }
    }
    return scale;
}



int cuda_betweenness_centrality (
    _IN_ int* V,
    _IN_ int* E,
    _IN_ double* W,
    _IN_ int* sources,
    _IN_ int len_V,
    _IN_ int len_E,
    _IN_ int len_sources,
    _IN_ int warp_size,
    _IN_ int is_directed,
    _IN_ int normalized,
    _IN_ int endpoints,
    _OUT_ double* BC
)
{
    int cuda_ret = cudaSuccess;
    int EG_ret = EG_GPU_SUCC;

    int block_size = 256;
//    size_t grid_size = len_sources / block_size + (len_sources % block_size != 0);
    size_t grid_size = len_V / block_size + (len_V % block_size != 0);
    size_t mem_free = 0, mem_total = 0;

    double scale = calc_scale(len_V, is_directed, normalized, endpoints);

    int *d_V = NULL, *d_E = NULL, *d_sources= NULL;
    int *d_U_2D = NULL, *d_F_2D = NULL, *d_st_2D = NULL, *d_st_idx_2D = NULL;
    double *d_W = NULL, *d_min_edge = NULL, *d_dist_2D = NULL, 
            *d_sigma_2D = NULL, *d_delta_2D = NULL, *d_BC = NULL;

    EXIT_IF_CUDA_FAILED(cudaMemGetInfo(&mem_free, &mem_total));
    while (true) {
        size_t mem_needed = sizeof(int) * len_V // d_V
                        + sizeof(int) * len_E // d_E
                        + sizeof(int) * len_sources // d_sources
                        + sizeof(int) * grid_size * len_V // d_U_2D
                        + sizeof(int) * grid_size * len_V // d_F_2D
                        + sizeof(int) * grid_size * len_V // d_st_2D
                        + sizeof(int) * grid_size * (len_V + 2) // d_st_idx_2D
                        + sizeof(double) * len_E // d_W
                        + sizeof(double) * len_V // d_min_edge
                        + sizeof(double) * grid_size * len_V // d_dist_2D
                        + sizeof(double) * grid_size * len_V // d_sigma_2D
                        + sizeof(double) * grid_size * len_V // d_delta_2D
                        + sizeof(double) * len_V // d_BC
                        ;
        if (mem_needed < mem_free / 2) {
            break;
        } else {
            grid_size /= 2;
        }
    }

    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_V, sizeof(int) * len_V));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_E, sizeof(int) * len_E));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_sources, sizeof(int) * len_sources));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_U_2D, sizeof(int) * grid_size * len_V));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_F_2D, sizeof(int) * grid_size * len_V));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_st_2D, sizeof(int) * grid_size * len_V));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_st_idx_2D, sizeof(int) * grid_size * (len_V + 2)));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_W, sizeof(double) * len_E));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_min_edge, sizeof(double) * len_V));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_dist_2D, sizeof(double) * grid_size * len_V));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_sigma_2D, sizeof(double) * grid_size * len_V));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_delta_2D, sizeof(double) * grid_size * len_V));
    EXIT_IF_CUDA_FAILED(cudaMalloc((void**)&d_BC, sizeof(double) * len_V));

    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_V, V, sizeof(int) * len_V, cudaMemcpyHostToDevice));
    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_E, E, sizeof(int) * len_E, cudaMemcpyHostToDevice));
    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_sources, sources, sizeof(int) * len_sources, cudaMemcpyHostToDevice));
    EXIT_IF_CUDA_FAILED(cudaMemcpy(d_W, W, sizeof(double) * len_E, cudaMemcpyHostToDevice));

    d_calc_min_edge<<<grid_size, block_size>>>(d_V, d_E, d_W, len_V, len_E, d_min_edge);

    d_dijkstra_bc<<<grid_size, block_size>>>(d_V, d_E, d_W, d_min_edge, d_sources, d_dist_2D, 
                                            d_sigma_2D, d_delta_2D, d_U_2D, d_F_2D, d_st_2D, 
                                            d_st_idx_2D, len_V, len_E, len_sources, warp_size,
                                            endpoints, d_BC);

    if (scale != 1.0) {
        d_rescale<<<grid_size, block_size>>>(len_V, scale, d_BC);
    }

    EXIT_IF_CUDA_FAILED(cudaMemcpy(BC, d_BC, sizeof(double) * len_V, cudaMemcpyDeviceToHost));

exit:
    cudaFree(d_V);
    cudaFree(d_E);
    cudaFree(d_sources);
    cudaFree(d_U_2D);
    cudaFree(d_F_2D);
    cudaFree(d_st_2D);
    cudaFree(d_st_idx_2D);
    cudaFree(d_W);
    cudaFree(d_min_edge);
    cudaFree(d_dist_2D);
    cudaFree(d_sigma_2D);
    cudaFree(d_delta_2D);
    cudaFree(d_BC);

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