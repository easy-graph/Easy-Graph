#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <assert.h>
#include <string.h>
#include <cuda_runtime.h>
#include <fstream>
#include <unordered_map>

using namespace std;

struct Edge {
    int u, v;
};

int N, M;
bool directed = false;
vector<int> h_csrRowPtr, h_csrColIdx;
vector<int> h_csrRowPtr_rev, h_csrColIdx_rev;

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if(err!=cudaSuccess){ \
        fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err)); \
        exit(1); \
    } \
}while(0)

void buildCSR(const vector<Edge>& edges, int N, bool directed,
             vector<int>& csrRowPtr, vector<int>& csrColIdx) {
    vector<Edge> sortedEdges;
    if (!directed) {
        sortedEdges.reserve(edges.size() * 2);
        for (auto &e : edges) {
            sortedEdges.push_back(e);
            sortedEdges.push_back({e.v, e.u});
        }
    }
    else {
        sortedEdges = edges;
    }

    sort(sortedEdges.begin(), sortedEdges.end(), [](const Edge& a, const Edge& b) {
        return (a.u < b.u) || (a.u == b.u && a.v < b.v);
    });

    csrRowPtr.assign(N + 1, 0);
    for (auto &e : sortedEdges) {
        csrRowPtr[e.u + 1]++;
    }
    for (int i = 1; i <= N; i++) {
        csrRowPtr[i] += csrRowPtr[i - 1];
    }
    csrColIdx.resize(sortedEdges.size());
    vector<int> offset(N, 0);
    for (auto &e : sortedEdges) {
        int idx = csrRowPtr[e.u] + offset[e.u]++;
        csrColIdx[idx] = e.v;
    }
}

void buildReverseGraphCSR(const vector<int>& csrRowPtr, const vector<int>& csrColIdx, int N,
                         vector<int>& csrRowPtr_rev, vector<int>& csrColIdx_rev) {
    vector<int> inDegree(N, 0);
    for (int u = 0; u < N; u++) {
        for (int i = csrRowPtr[u]; i < csrRowPtr[u + 1]; i++) {
            int v = csrColIdx[i];
            inDegree[v]++;
        }
    }
    csrRowPtr_rev.assign(N + 1, 0);
    for (int i = 0; i < N; i++) {
        csrRowPtr_rev[i + 1] = csrRowPtr_rev[i] + inDegree[i];
    }
    csrColIdx_rev.assign(csrRowPtr_rev[N], -1);

    vector<int> offset(N, 0);
    for (int u = 0; u < N; u++) {
        for (int i = csrRowPtr[u]; i < csrRowPtr[u + 1]; i++) {
            int v = csrColIdx[i];
            int pos = csrRowPtr_rev[v] + offset[v]++;
            csrColIdx_rev[pos] = u;
        }
    }
}

// ============ CPU ============
void cpu_cc(const vector<int>& csrRowPtr, const vector<int>& csrColIdx, int N, vector<int>& comp) {
    fill(comp.begin(), comp.end(), -1);
    int c_id = 0;
    for (int start = 0; start < N; start++) {
        if (comp[start] >= 0) continue;
        // BFS
        vector<int> q;
        q.push_back(start);
        comp[start] = c_id;
        for (int i = 0; i < (int)q.size(); i++) {
            int u = q[i];
            for (int e = csrRowPtr[u]; e < csrRowPtr[u + 1]; e++) {
                int v = csrColIdx[e];
                if (comp[v] < 0) {
                    comp[v] = c_id;
                    q.push_back(v);
                }
            }
        }
        c_id++;
    }
}

// CPU Kosaraju算法计算SCC
void cpu_dfs1(const vector<int>& csrRowPtr, const vector<int>& csrColIdx, int u, vector<bool>& vis, vector<int>& order) {
    vis[u] = true;
    for (int i = csrRowPtr[u]; i < csrRowPtr[u + 1]; i++) {
        int v = csrColIdx[i];
        if (!vis[v]) cpu_dfs1(csrRowPtr, csrColIdx, v, vis, order);
    }
    order.push_back(u);
}

void cpu_dfs2(const vector<int>& csrRowPtr_rev, const vector<int>& csrColIdx_rev, int u, vector<bool>& vis, vector<int>& comp, int cid) {
    vis[u] = true;
    comp[u] = cid;
    for (int i = csrRowPtr_rev[u]; i < csrRowPtr_rev[u + 1]; i++) {
        int v = csrColIdx_rev[i];
        if (!vis[v]) cpu_dfs2(csrRowPtr_rev, csrColIdx_rev, v, vis, comp, cid);
    }
}

void cpu_scc_kosaraju(const vector<int>& csrRowPtr, const vector<int>& csrColIdx,
                      const vector<int>& csrRowPtr_rev, const vector<int>& csrColIdx_rev,
                      int N, vector<int>& comp) {
    vector<bool> vis(N, false);
    vector<int> order;
    order.reserve(N);
    // 第一次DFS
    for (int i = 0; i < N; i++) {
        if (!vis[i]) cpu_dfs1(csrRowPtr, csrColIdx, i, vis, order);
    }

    fill(vis.begin(), vis.end(), false);
    fill(comp.begin(), comp.end(), -1);
    int cid = 0;
    for (int i = N - 1; i >= 0; i--) {
        int u = order[i];
        if (!vis[u]) {
            cpu_dfs2(csrRowPtr_rev, csrColIdx_rev, u, vis, comp, cid);
            cid++;
        }
    }
}

// ============ GPU ============
__global__ void gpu_init_array(int *d_arr, int val, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) d_arr[idx] = val;
}

// 使用atomicMin找最小编号未访问节点
__global__ void gpu_find_unvisited_cc(int *d_component, int *Nptr, int *d_unvisited_node) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = *Nptr;
    if (idx < N && d_component[idx] == -1) {
        atomicMin(d_unvisited_node, idx);
    }
}

__global__ void gpu_cc_expand(const int *d_csrRowPtr, const int *d_csrColIdx, int *Nptr, int *d_component, int current_label,
                              int *frontier, int front_size, int *d_next_frontier, int *d_next_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = *Nptr;
    if (idx < front_size) {
        int u = frontier[idx];
        int start = d_csrRowPtr[u], end = d_csrRowPtr[u + 1];
        for (int i = start; i < end; i++) {
            int v = d_csrColIdx[i];
            if (v < N && atomicCAS(&d_component[v], -1, current_label) == -1) {
                int pos = atomicAdd(d_next_size, 1);
                d_next_frontier[pos] = v;
            }
        }
    }
}

// 无向图连通分量GPU计算
void gpu_connected_components(const vector<int>& csrRowPtr, const vector<int>& csrColIdx, int N, vector<int>& component) {
    int *d_csrRowPtr, *d_csrColIdx;
    int *d_component;
    int *d_N, *d_unvisited_node;

    CUDA_CHECK(cudaMalloc(&d_csrRowPtr, (N + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_csrColIdx, csrColIdx.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_component, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_N, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_unvisited_node, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_csrRowPtr, csrRowPtr.data(), (N + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_csrColIdx, csrColIdx.data(), csrColIdx.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_N, &N, sizeof(int), cudaMemcpyHostToDevice));

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    gpu_init_array<<<gridSize, blockSize>>>(d_component, -1, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int *d_frontier, *d_next_frontier;
    CUDA_CHECK(cudaMalloc(&d_frontier, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_next_frontier, N * sizeof(int)));
    int *d_next_size;
    CUDA_CHECK(cudaMalloc(&d_next_size, sizeof(int)));

    int current_label = 0;
    while (true) {
        int init_val = N; // 初始值设为N(大于最大节点编号)
        CUDA_CHECK(cudaMemcpy(d_unvisited_node, &init_val, sizeof(int), cudaMemcpyHostToDevice));

        gpu_find_unvisited_cc<<<gridSize, blockSize>>>(d_component, d_N, d_unvisited_node);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        int unvisited_node;
        CUDA_CHECK(cudaMemcpy(&unvisited_node, d_unvisited_node, sizeof(int), cudaMemcpyDeviceToHost));

        if (unvisited_node == N) break; // 无未访问节点

        // 将该节点标记到当前分量
        CUDA_CHECK(cudaMemcpy(d_component + unvisited_node, &current_label, sizeof(int), cudaMemcpyHostToDevice));

        int front_size = 1;
        CUDA_CHECK(cudaMemcpy(d_frontier, &unvisited_node, sizeof(int), cudaMemcpyHostToDevice));

        while (front_size > 0) {
            int next_size = 0;
            CUDA_CHECK(cudaMemcpy(d_next_size, &next_size, sizeof(int), cudaMemcpyHostToDevice));

            int gridSizeF = (front_size + blockSize - 1) / blockSize;
            gpu_cc_expand<<<gridSizeF, blockSize>>>(d_csrRowPtr, d_csrColIdx, d_N, d_component, current_label,
                                                    d_frontier, front_size, d_next_frontier, d_next_size);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaMemcpy(&next_size, d_next_size, sizeof(int), cudaMemcpyDeviceToHost));
            if (next_size == 0) break;

            // swap
            int *temp = d_frontier; d_frontier = d_next_frontier; d_next_frontier = temp;
            front_size = next_size;
        }

        CUDA_CHECK(cudaDeviceSynchronize());
        current_label++;
    }

    CUDA_CHECK(cudaMemcpy(component.data(), d_component, N * sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_csrRowPtr));
    CUDA_CHECK(cudaFree(d_csrColIdx));
    CUDA_CHECK(cudaFree(d_component));
    CUDA_CHECK(cudaFree(d_N));
    CUDA_CHECK(cudaFree(d_unvisited_node));
    CUDA_CHECK(cudaFree(d_frontier));
    CUDA_CHECK(cudaFree(d_next_frontier));
    CUDA_CHECK(cudaFree(d_next_size));
}

// ========== SCC/WCC 前向后向算法使用前沿BFS ==========
__global__ void gpu_bfs_frontier_expand(const int *d_csrRowPtr, const int *d_csrColIdx,
                                        int *d_visited, int *frontier, int front_size,
                                        int *d_next_frontier, int *d_next_size, const int *d_valid, int current_level) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < front_size) {
        int u = frontier[idx];
        int start = d_csrRowPtr[u], end = d_csrRowPtr[u + 1];
        for (int i = start; i < end; i++) {
            int v = d_csrColIdx[i];
            if (d_valid[v] == 1 && atomicCAS(&d_visited[v], -1, current_level + 1) == -1) {
                int pos = atomicAdd(d_next_size, 1);
                d_next_frontier[pos] = v;
            }
        }
    }
}

void gpu_run_frontier_bfs(const int *d_csrRowPtr, const int *d_csrColIdx, int N, int pivot, const int *d_valid, int *d_visited,
                          int *d_frontier, int *d_next_frontier) {
    int zero = 0;
    CUDA_CHECK(cudaMemcpy(d_visited + pivot, &zero, sizeof(int), cudaMemcpyHostToDevice));

    int front_size = 1;
    CUDA_CHECK(cudaMemcpy(d_frontier, &pivot, sizeof(int), cudaMemcpyHostToDevice));

    int *d_next_size;
    CUDA_CHECK(cudaMalloc(&d_next_size, sizeof(int)));

    int level = 0;
    int blockSize = 256;
    while (front_size > 0) {
        int next_size = 0;
        CUDA_CHECK(cudaMemcpy(d_next_size, &next_size, sizeof(int), cudaMemcpyHostToDevice));

        int gridSize = (front_size + blockSize - 1) / blockSize;
        gpu_bfs_frontier_expand<<<gridSize, blockSize>>>(d_csrRowPtr, d_csrColIdx, d_visited, d_frontier, front_size,
                                                         d_next_frontier, d_next_size, d_valid, level);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(&next_size, d_next_size, sizeof(int), cudaMemcpyDeviceToHost));
        if (next_size == 0) {
            break;
        }
        int *temp = d_frontier;
        d_frontier = d_next_frontier;
        d_next_frontier = temp;
        front_size = next_size;
        level++;
    }

    CUDA_CHECK(cudaFree(d_next_size));
}

__global__ void mark_valid_nodes(int *d_valid, int val, int N) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u < N) d_valid[u] = val;
}

__global__ void gpu_check_unvisited_scc(int *d_valid, int N, int *d_found, int *d_pivot) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u < N && d_valid[u] == 1 && atomicCAS(d_found, 0, 1) == 0) {
        *d_pivot = u;
    }
}

__global__ void gpu_intersect_and_assign(int *d_forward, int *d_backward, int *d_valid, int *d_component, int N, int comp_id) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u < N && d_valid[u] == 1 && d_forward[u] >= 0 && d_backward[u] >= 0) {
        d_component[u] = comp_id;
        d_valid[u] = 0; // 已分配
    }
}

void gpu_scc_forward_backward(const vector<int>& csrRowPtr, const vector<int>& csrColIdx,
                              const vector<int>& csrRowPtr_rev, const vector<int>& csrColIdx_rev, int N, vector<int>& component) {
    int *d_csrRowPtr, *d_csrColIdx;
    int *d_csrRowPtr_rev, *d_csrColIdx_rev;
    CUDA_CHECK(cudaMalloc(&d_csrRowPtr, (N + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_csrColIdx, csrColIdx.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_csrRowPtr_rev, (N + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_csrColIdx_rev, csrColIdx_rev.size() * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_csrRowPtr, csrRowPtr.data(), (N + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_csrColIdx, csrColIdx.data(), csrColIdx.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_csrRowPtr_rev, csrRowPtr_rev.data(), (N + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_csrColIdx_rev, csrColIdx_rev.data(), csrColIdx_rev.size() * sizeof(int), cudaMemcpyHostToDevice));

    int *d_forward, *d_backward, *d_component, *d_valid;
    CUDA_CHECK(cudaMalloc(&d_forward, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_backward, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_component, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_valid, N * sizeof(int)));

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    gpu_init_array<<<gridSize, blockSize>>>(d_component, -1, N);
    CUDA_CHECK(cudaGetLastError());
    mark_valid_nodes<<<gridSize, blockSize>>>(d_valid, 1, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int *d_found, *d_pivot;
    CUDA_CHECK(cudaMalloc(&d_found, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pivot, sizeof(int)));

    int *d_frontier, *d_next_frontier;
    CUDA_CHECK(cudaMalloc(&d_frontier, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_next_frontier, N * sizeof(int)));

    int comp_id = 0;

    while (true) {
        int found = 0;
        CUDA_CHECK(cudaMemcpy(d_found, &found, sizeof(int), cudaMemcpyHostToDevice));
        gpu_check_unvisited_scc<<<gridSize, blockSize>>>(d_valid, N, d_found, d_pivot);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&found, d_found, sizeof(int), cudaMemcpyDeviceToHost));
        if (!found) break;

        int pivot;
        CUDA_CHECK(cudaMemcpy(&pivot, d_pivot, sizeof(int), cudaMemcpyDeviceToHost));

        // 不要在这里修改 d_valid[pivot]
        // 原代码中通过 gpu_intersect_and_assign 进行修改

        // 前向 BFS
        gpu_init_array<<<gridSize, blockSize>>>(d_forward, -1, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        gpu_run_frontier_bfs(d_csrRowPtr, d_csrColIdx, N, pivot, d_valid, d_forward,
                             d_frontier, d_next_frontier);
        CUDA_CHECK(cudaDeviceSynchronize());

        // 后向 BFS
        gpu_init_array<<<gridSize, blockSize>>>(d_backward, -1, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        gpu_run_frontier_bfs(d_csrRowPtr_rev, d_csrColIdx_rev, N, pivot, d_valid, d_backward,
                             d_frontier, d_next_frontier);
        CUDA_CHECK(cudaDeviceSynchronize());

        // 交集并分配组件 ID
        gpu_intersect_and_assign<<<gridSize, blockSize>>>(d_forward, d_backward, d_valid, d_component, N, comp_id);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        comp_id++;
    }

    CUDA_CHECK(cudaMemcpy(component.data(), d_component, N * sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_csrRowPtr));
    CUDA_CHECK(cudaFree(d_csrColIdx));
    CUDA_CHECK(cudaFree(d_csrRowPtr_rev));
    CUDA_CHECK(cudaFree(d_csrColIdx_rev));
    CUDA_CHECK(cudaFree(d_forward));
    CUDA_CHECK(cudaFree(d_backward));
    CUDA_CHECK(cudaFree(d_component));
    CUDA_CHECK(cudaFree(d_valid));
    CUDA_CHECK(cudaFree(d_found));
    CUDA_CHECK(cudaFree(d_pivot));
    CUDA_CHECK(cudaFree(d_frontier));
    CUDA_CHECK(cudaFree(d_next_frontier));
}

// 将分量输出到文件的辅助函数
void write_components_to_file(const string &filename, const vector<int> &comp, const string &title) {
    ofstream fout(filename, ios::app);
    if (!fout) {
        cerr << "Cannot open " << filename << " for writing!" << endl;
        return;
    }

    // 检查是否有未分配的组件
    bool has_unassigned = false;
    for (size_t i = 0; i < comp.size(); i++) {
        if (comp[i] == -1) {
            has_unassigned = true;
            break;
        }
    }

    if (has_unassigned) {
        cout << "Warning: There are unassigned nodes in the components." << endl;
    }

    // 计算最大组件ID，确保不包含-1
    int max_comp = -1;
    for (size_t i = 0; i < comp.size(); i++) {
        if (comp[i] > max_comp) max_comp = comp[i];
    }

    // 如果有未分配的节点，将max_comp加1
    if (has_unassigned) {
        max_comp++;
    }

    vector<vector<int>> comps(max_comp + 1);
    for (int i = 0; i < (int)comp.size(); i++) {
        if (comp[i] == -1) {
            comps[max_comp].push_back(i); // 将未分配的节点放入最后一个组
        }
        else {
            comps[comp[i]].push_back(i);
        }
    }

    fout << title << endl;
    for (int c = 0; c <= max_comp; c++) {
        fout << "{";
        for (size_t idx = 0; idx < comps[c].size(); idx++) {
            fout << comps[c][idx];
            if (idx + 1 < comps[c].size()) fout << ", ";
        }
        fout << "}" << endl;
    }
    fout << endl;

    fout.close();
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s input_graph.txt directed/undirected\n", argv[0]);
        return 1;
    }

    string filename = argv[1];
    string type = argv[2];
    directed = (type == "directed");

    FILE *f = fopen(filename.c_str(), "r");
    if (!f) {
        fprintf(stderr, "Cannot open file %s\n", filename.c_str());
        return 1;
    }

    vector<Edge> edges;
    vector<int> node_ids;
    int u, v;
    // 读取所有边，并记录所有节点id
    while (fscanf(f, "%d %d", &u, &v) == 2) {
        edges.push_back({u, v});
        node_ids.push_back(u);
        node_ids.push_back(v);
    }
    fclose(f);

    if (edges.empty()) {
        printf("Empty graph!\n");
        return 0;
    }

    sort(node_ids.begin(), node_ids.end());
    node_ids.erase(unique(node_ids.begin(), node_ids.end()), node_ids.end());

    auto compress = [&](int x) -> int {
        auto it = std::lower_bound(node_ids.begin(), node_ids.end(), x);
        return (int)(it - node_ids.begin());
    };

    N = (int)node_ids.size();
    M = (int)edges.size();

    for (auto &e : edges) {
        e.u = compress(e.u);
        e.v = compress(e.v);
    }

    buildCSR(edges, N, directed, h_csrRowPtr, h_csrColIdx);
    if (directed) {
        buildReverseGraphCSR(h_csrRowPtr, h_csrColIdx, N, h_csrRowPtr_rev, h_csrColIdx_rev);
    }

    vector<int> gpu_component(N, -1);
    vector<int> cpu_component_res(N, -1);

    // CPU计算
    auto start_cpu = chrono::high_resolution_clock::now();
    if (!directed) {
        cpu_cc(h_csrRowPtr, h_csrColIdx, N, cpu_component_res);
    }
    else {
        cpu_scc_kosaraju(h_csrRowPtr, h_csrColIdx, h_csrRowPtr_rev, h_csrColIdx_rev, N, cpu_component_res);
    }
    auto end_cpu = chrono::high_resolution_clock::now();
    double cpu_time = chrono::duration<double>(end_cpu - start_cpu).count();

    // GPU计算
    auto start_gpu = chrono::high_resolution_clock::now();
    if (!directed) {
        gpu_connected_components(h_csrRowPtr, h_csrColIdx, N, gpu_component);
    }
    else {
        gpu_scc_forward_backward(h_csrRowPtr, h_csrColIdx, h_csrRowPtr_rev, h_csrColIdx_rev, N, gpu_component);
    }
    auto end_gpu = chrono::high_resolution_clock::now();
    double gpu_time = chrono::duration<double>(end_gpu - start_gpu).count();

    cout << "CPU Time: " << cpu_time << " s" << endl;
    cout << "GPU Time: " << gpu_time << " s" << endl;

    int max_cpu_comp = *max_element(cpu_component_res.begin(), cpu_component_res.end());
    int max_gpu_comp = *max_element(gpu_component.begin(), gpu_component.end());
    vector<int> count_cpu(max_cpu_comp + 1, 0), count_gpu(max_gpu_comp + 1, 0);
    for (int i = 0; i < N; i++) {
        if (cpu_component_res[i] >= 0 && cpu_component_res[i] < (int)count_cpu.size()) {
            count_cpu[cpu_component_res[i]]++;
        }
        else {
            // Handle unexpected component IDs
            // 可以选择忽略或统计到一个特殊的计数器
        }

        if (gpu_component[i] >= 0 && gpu_component[i] < (int)count_gpu.size()) {
            count_gpu[gpu_component[i]]++;
        }
        else {
            // Handle unexpected component IDs
            // 可以选择忽略或统计到一个特殊的计数器
        }
    }
    sort(count_cpu.begin(), count_cpu.end());
    sort(count_gpu.begin(), count_gpu.end());

    bool consistent = (max_cpu_comp == max_gpu_comp && count_cpu == count_gpu);

    cout << "CPU components count: " << (max_cpu_comp + 1) << endl;
    cout << "GPU components count: " << (max_gpu_comp + 1) << endl;
    cout << "Result consistent: " << (consistent ? "Yes" : "No") << endl;

    // 输出结果到文件
    if (!directed) {
        // 无向图：输出CC到cc.txt
        {
            ofstream fout("cc.txt");
            fout.close();
        }
        write_components_to_file("cc.txt", cpu_component_res, "CPU CC Components");
        write_components_to_file("cc.txt", gpu_component, "GPU CC Components");
    }
    else {
        // 有向图：输出SCC到scc.txt
        {
            ofstream fout("scc.txt");
            fout.close();
        }
        write_components_to_file("scc.txt", cpu_component_res, "CPU SCC Components");
        write_components_to_file("scc.txt", gpu_component, "GPU SCC Components");

        // 计算WCC
        vector<int> wcc_csrRowPtr, wcc_csrColIdx;
        buildCSR(edges, N, false, wcc_csrRowPtr, wcc_csrColIdx);
        vector<int> cpu_wcc(N, -1), gpu_wcc(N, -1);
        cpu_cc(wcc_csrRowPtr, wcc_csrColIdx, N, cpu_wcc);
        gpu_connected_components(wcc_csrRowPtr, wcc_csrColIdx, N, gpu_wcc);

        {
            ofstream fout("wcc.txt");
            fout.close();
        }
        write_components_to_file("wcc.txt", cpu_wcc, "CPU WCC Components");
        write_components_to_file("wcc.txt", gpu_wcc, "GPU WCC Components");
    }

    return 0;
}