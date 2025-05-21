#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>
#include <string>

using namespace std;

inline void checkCudaError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        cerr << "CUDA error in file " << file << " at line " << line << ": " 
             << cudaGetErrorString(err) << endl;
        exit(EXIT_FAILURE);
    }
}

#define CHECK_CUDA_ERROR(err) checkCudaError(err, __FILE__, __LINE__)

inline std::string trim(const std::string& s) {
    auto start = s.begin();
    while (start != s.end() && std::isspace(*start)) start++;
    auto end = s.end();
    if (start == end) return "";
    do {
        end--;
    } while (distance(start, end) > 0 && std::isspace(*end));
    return std::string(start, end + 1);
}

__global__ void directed_clustering_coefficient_csr_kernel(
    const int* __restrict__ row_ptr_out,
    const int* __restrict__ col_indices_out,
    const int* __restrict__ row_ptr_in,
    const int* __restrict__ col_indices_in,
    const int* __restrict__ degree_out,
    const int* __restrict__ degree_in,
    double* cc,
    int num_nodes
)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= num_nodes) return;

    int in_deg = degree_in[u];
    int out_deg = degree_out[u];
    int dtotal = in_deg + out_deg;

    int reciprocal = 0;
    int start_out = row_ptr_out[u];
    int end_out = row_ptr_out[u + 1];
    int start_in_u = row_ptr_in[u];
    int end_in_u = row_ptr_in[u + 1];

    for(int i = start_out; i < end_out; ++i){
        int v = col_indices_out[i];
        int left = start_in_u;
        int right = end_in_u - 1;
        bool found = false;
        while(left <= right){
            int mid = left + (right - left) / 2;
            if(col_indices_in[mid] == v){
                found = true;
                break;
            }
            else if(col_indices_in[mid] < v){
                left = mid + 1;
            }
            else{
                right = mid - 1;
            }
        }
        if(found){
            reciprocal++;
        }
    }

    double denom = (double)(dtotal * (dtotal - 1) - 2.0 * reciprocal);
    if(denom <= 0){
        cc[u] = 0.0;
        return;
    }

    int directed_triangles = 0;

    int start_in_edges = row_ptr_in[u];
    int end_in_edges = row_ptr_in[u + 1];
    int start_out_edges = row_ptr_out[u];
    int end_out_edges = row_ptr_out[u + 1];

    for(int i = start_in_edges; i < end_in_edges; ++i){
        int v = col_indices_in[i];
        int start_pred_v = row_ptr_in[v];
        int end_pred_v = row_ptr_in[v + 1];
        for(int j = start_pred_v; j < end_pred_v; ++j){
            int k = col_indices_in[j];
            if(k == u) continue;
            int left = start_out_edges;
            int right = end_out_edges - 1;
            bool exists = false;
            while(left <= right){
                int mid = left + (right - left) / 2;
                if(col_indices_out[mid] == k){
                    exists = true;
                    break;
                }
                else if(col_indices_out[mid] < k){
                    left = mid + 1;
                }
                else{
                    right = mid - 1;
                }
            }
            if(exists){
                directed_triangles++;
            }
        }

        int start_adj_out_v = row_ptr_out[v];
        int end_adj_out_v = row_ptr_out[v + 1];
        for(int j = start_adj_out_v; j < end_adj_out_v; ++j){
            int k = col_indices_out[j];
            if(k == u) continue;
            int left = start_in_edges;
            int right = end_in_edges - 1;
            bool exists = false;
            while(left <= right){
                int mid = left + (right - left) / 2;
                if(col_indices_in[mid] == k){
                    exists = true;
                    break;
                }
                else if(col_indices_in[mid] < k){
                    left = mid + 1;
                }
                else{
                    right = mid - 1;
                }
            }
            if(exists){
                directed_triangles++;
            }
        }
    }

    for(int i = start_out_edges; i < end_out_edges; ++i){
        int v = col_indices_out[i];
        int start_pred_v = row_ptr_in[v];
        int end_pred_v = row_ptr_in[v + 1];
        for(int j = start_pred_v; j < end_pred_v; ++j){
            int k = col_indices_in[j];
            if(k == u) continue;
            int left = start_out_edges;
            int right = end_out_edges - 1;
            bool exists = false;
            while(left <= right){
                int mid = left + (right - left) / 2;
                if(col_indices_out[mid] == k){
                    exists = true;
                    break;
                }
                else if(col_indices_out[mid] < k){
                    left = mid + 1;
                }
                else{
                    right = mid - 1;
                }
            }
            if(exists){
                directed_triangles++;
            }
        }

        int start_adj_out_v = row_ptr_out[v];
        int end_adj_out_v = row_ptr_out[v + 1];
        for(int j = start_adj_out_v; j < end_adj_out_v; ++j){
            int k = col_indices_out[j];
            if(k == u) continue;
            int left = start_in_edges;
            int right = end_in_edges - 1;
            bool exists = false;
            while(left <= right){
                int mid = left + (right - left) / 2;
                if(col_indices_in[mid] == k){
                    exists = true;
                    break;
                }
                else if(col_indices_in[mid] < k){
                    left = mid + 1;
                }
                else{
                    right = mid - 1;
                }
            }
            if(exists){
                directed_triangles++;
            }
        }
    }

    double t = (double)directed_triangles / 2.0;

    double coeff = (denom != 0.0 && t != 0.0) ? (t / denom) : 0.0;
    cc[u] = coeff;
}

__global__ void undirected_clustering_coefficient_csr_kernel(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_indices,
    const int* __restrict__ degree,
    double* cc,
    int num_nodes
)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= num_nodes) return;

    int d = degree[u];
    if (d < 2) {
        cc[u] = 0.0;
        return;
    }

    int start_u = row_ptr[u];
    int end_u = row_ptr[u + 1];

    int triangles = 0;
    for(int m = 0; m < d; m++) {
        int neighbor_i = col_indices[start_u + m];
        for(int n = m + 1; n < d; n++) {
            int neighbor_j = col_indices[start_u + n];
            
            int start_v = row_ptr[neighbor_i];
            int end_v = row_ptr[neighbor_i + 1];
            int left = start_v;
            int right = end_v - 1;
            bool found = false;
            while(left <= right){
                int mid = left + (right - left) / 2;
                if(col_indices[mid] == neighbor_j){
                    found = true;
                    break;
                }
                else if(col_indices[mid] < neighbor_j){
                    left = mid + 1;
                }
                else{
                    right = mid - 1;
                }
            }
            if(found){
                triangles++;
            }
        }
    }

    double possible = (static_cast<double>(d) * (d - 1)) / 2.0;
    cc[u] = static_cast<double>(triangles) / possible;
}

void host_clustering_coefficient(const unordered_map<int, unordered_set<int>>& adj_out,
                                 const unordered_map<int, unordered_set<int>>& adj_in,
                                 unordered_map<int, double>& cc, bool is_directed) {
    for (const auto& pair : adj_out) {
        int u = pair.first;
        const unordered_set<int>& out_neighbors = pair.second;
        const unordered_set<int>& in_neighbors = adj_in.at(u);
        int in_deg = in_neighbors.size();
        int out_deg = out_neighbors.size();
        int dtotal = in_deg + out_deg;

        if(!is_directed){
            const unordered_set<int>& neighbors = adj_out.at(u);
            int d = neighbors.size();
            if(d < 2){
                cc[u] = 0.0;
                continue;
            }
            int triangles = 0;
            for(auto it1 = neighbors.begin(); it1 != neighbors.end(); ++it1){
                auto it2 = it1; 
                for(++it2; it2 != neighbors.end(); ++it2){
                    if(adj_out.at(*it1).count(*it2)){
                        triangles++;
                    }
                }
            }
            double coeff = (static_cast<double>(triangles) / (static_cast<double>(d) * (d - 1) / 2.0));
            cc[u] = coeff;
        }
        else{
            if(dtotal < 2){
                cc[u] = 0.0;
                continue;
            }
            int reciprocal = 0;
            for(auto it = out_neighbors.begin(); it != out_neighbors.end(); ++it){
                if(in_neighbors.find(*it) != in_neighbors.end()){
                    reciprocal++;
                }
            }

            int directed_triangles = 0;
            for(auto &v : in_neighbors){
                if(adj_in.find(v) != adj_in.end()){
                    for(auto &k : adj_in.at(v)){
                        if(k == u) continue;
                        if(out_neighbors.find(k) != out_neighbors.end()){
                            directed_triangles++;
                        }
                    }
                }

                if(adj_out.find(v) != adj_out.end()){
                    for(auto &k : adj_out.at(v)){
                        if(k == u) continue;
                        if(in_neighbors.find(k) != in_neighbors.end()){
                            directed_triangles++;
                        }
                    }
                }
            }

            for(auto &v : out_neighbors){
                if(adj_in.find(v) != adj_in.end()){
                    for(auto &k : adj_in.at(v)){
                        if(k == u) continue;
                        if(out_neighbors.find(k) != out_neighbors.end()){
                            directed_triangles++;
                        }
                    }
                }

                if(adj_out.find(v) != adj_out.end()){
                    for(auto &k : adj_out.at(v)){
                        if(k == u) continue;
                        if(in_neighbors.find(k) != in_neighbors.end()){
                            directed_triangles++;
                        }
                    }
                }
            }

            double t = static_cast<double>(directed_triangles) / 2.0;

            double denom = (static_cast<double>(dtotal) * (dtotal - 1) - 2.0 * reciprocal);
            double coeff = (denom != 0.0 && t != 0.0) ? (t / denom) : 0.0;
            cc[u] = coeff;
        }
    }
}

int main(int argc, char* argv[]) {
    auto total_start_time = chrono::high_resolution_clock::now();

    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <txt file path> <undirected\\directed>\n";
        return 1;
    }

    string input_file = argv[1];
    string graph_type = argv[2];

    bool is_directed = false;
    if(graph_type == "directed"){
        is_directed = true;
    }
    else if(graph_type == "undirected"){
        is_directed = false;
    }
    else{
        cerr << " directed or undirected\n";
        return 1;
    }

    ifstream infile(input_file);
    if (!infile.is_open()) {
        cerr << "Cannot open file: " << input_file << "\n";
        return 1;
    }

    unordered_map<int, unordered_set<int>> adj_out;
    unordered_map<int, unordered_set<int>> adj_in;

    string line;
    while(getline(infile, line)){
        line = trim(line);
        if(line.empty()) continue;
        int u, v;
        stringstream ss(line);
        if(!(ss >> u >> v)){
            cerr << "无效的行格式: " << line << "\n";
            continue;
        }
        if(u == v){
            continue; 
        }
        adj_out[u].insert(v);
        adj_out[v]; 
        if(!is_directed){
            adj_out[v].insert(u);
            adj_in[u].insert(v);
            adj_in[v].insert(u);
        }
        else{
            adj_in[v].insert(u);
            adj_in[u];
        }
    }
    infile.close();

    unordered_map<int, int> node_id_to_idx;
    vector<int> idx_to_node_id;
    int idx_counter = 0;
    for(auto &pair : adj_out){
        if(node_id_to_idx.find(pair.first) == node_id_to_idx.end()){
            node_id_to_idx[pair.first] = idx_counter;
            idx_to_node_id.push_back(pair.first);
            idx_counter++;
        }
    }
    for(auto &pair : adj_in){
        if(node_id_to_idx.find(pair.first) == node_id_to_idx.end()){
            node_id_to_idx[pair.first] = idx_counter;
            idx_to_node_id.push_back(pair.first);
            idx_counter++;
        }
    }
    int num_nodes = node_id_to_idx.size();

    vector<int> row_ptr_out(num_nodes + 1, 0);
    vector<int> col_indices_out;
    col_indices_out.reserve(adj_out.size() * 10);

    for(int i = 0; i < num_nodes; i++) {
        int node_id = idx_to_node_id[i];
        if(adj_out.find(node_id) != adj_out.end()){
            vector<int> sorted_neighbors;
            for(auto &neighbor_id : adj_out[node_id]){
                if(node_id_to_idx.find(neighbor_id) != node_id_to_idx.end()){
                    int neighbor_idx = node_id_to_idx[neighbor_id];
                    sorted_neighbors.push_back(neighbor_idx);
                }
            }
            sort(sorted_neighbors.begin(), sorted_neighbors.end());
            col_indices_out.insert(col_indices_out.end(), sorted_neighbors.begin(), sorted_neighbors.end());
            row_ptr_out[i + 1] = col_indices_out.size();
        }
        else{
            row_ptr_out[i + 1] = row_ptr_out[i];
        }
    }

    vector<int> row_ptr_in(num_nodes + 1, 0);
    vector<int> col_indices_in;
    if(is_directed){
        col_indices_in.reserve(adj_in.size() * 10);
        for(int i = 0; i < num_nodes; i++) {
            int node_id = idx_to_node_id[i];
            if(adj_in.find(node_id) != adj_in.end()){
                vector<int> sorted_neighbors;
                for(auto &neighbor_id : adj_in[node_id]){
                    if(node_id_to_idx.find(neighbor_id) != node_id_to_idx.end()){
                        int neighbor_idx = node_id_to_idx[neighbor_id];
                        sorted_neighbors.push_back(neighbor_idx);
                    }
                }
                sort(sorted_neighbors.begin(), sorted_neighbors.end());
                col_indices_in.insert(col_indices_in.end(), sorted_neighbors.begin(), sorted_neighbors.end());
                row_ptr_in[i + 1] = col_indices_in.size();
            }
            else{
                row_ptr_in[i + 1] = row_ptr_in[i];
            }
        }
    }

    int num_edges_out = col_indices_out.size();
    int num_edges_in = is_directed ? col_indices_in.size() : 0;

    vector<int> degrees_out(num_nodes, 0);
    vector<int> degrees_in(num_nodes, 0);
    for(int i = 0; i < num_nodes; i++) {
        degrees_out[i] = row_ptr_out[i + 1] - row_ptr_out[i];
        if(is_directed){
            degrees_in[i] = row_ptr_in[i + 1] - row_ptr_in[i];
        }
    }

    unordered_map<int, double> cc_host;
    auto host_start_time = chrono::high_resolution_clock::now();
    host_clustering_coefficient(adj_out, adj_in, cc_host, is_directed);
    auto host_end_time = chrono::high_resolution_clock::now();
    auto host_duration = chrono::duration_cast<chrono::duration<double>>(host_end_time - host_start_time);
    cout << "Host execution time: " << fixed << setprecision(6) << host_duration.count() << " s\n";

    int* d_row_ptr_out = nullptr;
    int* d_col_indices_out = nullptr;
    int* d_row_ptr_in = nullptr;
    int* d_col_indices_in = nullptr;
    int* d_degree_out = nullptr;
    int* d_degree_in = nullptr;
    double* d_cc = nullptr;

    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_row_ptr_out, (num_nodes + 1) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_col_indices_out, num_edges_out * sizeof(int)));
    if(is_directed){
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_row_ptr_in, (num_nodes + 1) * sizeof(int)));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_col_indices_in, num_edges_in * sizeof(int)));
    }
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_degree_out, num_nodes * sizeof(int)));
    if(is_directed){
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_degree_in, num_nodes * sizeof(int)));
    }
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_cc, num_nodes * sizeof(double)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_row_ptr_out, row_ptr_out.data(), (num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_col_indices_out, col_indices_out.data(), num_edges_out * sizeof(int), cudaMemcpyHostToDevice));
    if(is_directed){
        CHECK_CUDA_ERROR(cudaMemcpy(d_row_ptr_in, row_ptr_in.data(), (num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_col_indices_in, col_indices_in.data(), num_edges_in * sizeof(int), cudaMemcpyHostToDevice));
    }
    CHECK_CUDA_ERROR(cudaMemcpy(d_degree_out, degrees_out.data(), num_nodes * sizeof(int), cudaMemcpyHostToDevice));
    if(is_directed){
        CHECK_CUDA_ERROR(cudaMemcpy(d_degree_in, degrees_in.data(), num_nodes * sizeof(int), cudaMemcpyHostToDevice));
    }

    auto gpu_start_time = chrono::high_resolution_clock::now();
    int threads_per_block = 256;
    int blocks = (num_nodes + threads_per_block - 1) / threads_per_block;

    if(is_directed){
        directed_clustering_coefficient_csr_kernel<<<blocks, threads_per_block>>>(
            d_row_ptr_out,
            d_col_indices_out,
            d_row_ptr_in,
            d_col_indices_in,
            d_degree_out,
            d_degree_in,
            d_cc,
            num_nodes
        );
    }
    else{
        undirected_clustering_coefficient_csr_kernel<<<blocks, threads_per_block>>>(
            d_row_ptr_out,
            d_col_indices_out,
            d_degree_out,
            d_cc,
            num_nodes
        );
    }
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    auto gpu_end_time = chrono::high_resolution_clock::now();
    auto gpu_duration = chrono::duration_cast<chrono::duration<double>>(gpu_end_time - gpu_start_time);
    cout << "GPU execution time: " << fixed << setprecision(6) << gpu_duration.count() << " s\n";

    vector<double> cc_gpu(num_nodes, 0.0);
    CHECK_CUDA_ERROR(cudaMemcpy(cc_gpu.data(), d_cc, num_nodes * sizeof(double), cudaMemcpyDeviceToHost));

    #include <fstream>   // For file output
#include <iomanip>   // For std::setw, std::fixed, std::setprecision

// 打开文件进行写入
std::ofstream outFile("output.txt");

if (outFile.is_open()) {
    // 输出表头到文件
    outFile << "\nNode ID | CPU | GPU\n";
    outFile << "------------------------------------------\n";

    // 输出每个节点的信息到文件
    for (int i = 0; i < num_nodes; i++) {
        int node_id = idx_to_node_id[i];
        double cpu_cc = (cc_host.find(node_id) != cc_host.end()) ? cc_host[node_id] : 0.0;
        double gpu_cc = cc_gpu[i];

        // 将内容写入文件
        outFile << std::setw(7) << node_id
                << " | " 
                << std::setw(12) << std::fixed << std::setprecision(6) << cpu_cc
                << " | "
                << std::setw(12) << std::fixed << std::setprecision(6) << gpu_cc << "\n";
    }

    // 关闭文件
    outFile.close();
} 

    CHECK_CUDA_ERROR(cudaFree(d_row_ptr_out));
    CHECK_CUDA_ERROR(cudaFree(d_col_indices_out));
    if(is_directed){
        CHECK_CUDA_ERROR(cudaFree(d_row_ptr_in));
        CHECK_CUDA_ERROR(cudaFree(d_col_indices_in));
    }
    CHECK_CUDA_ERROR(cudaFree(d_degree_out));
    if(is_directed){
        CHECK_CUDA_ERROR(cudaFree(d_degree_in));
    }
    CHECK_CUDA_ERROR(cudaFree(d_cc));

    auto total_end_time = chrono::high_resolution_clock::now();
    auto total_duration = chrono::duration_cast<chrono::duration<double>>(total_end_time - total_start_time);
    cout << "\nExec: " << fixed << setprecision(6) << total_duration.count() << " s\n";

    return 0;
}
