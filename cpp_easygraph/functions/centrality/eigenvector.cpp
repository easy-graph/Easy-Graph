#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "../../classes/graph.h"
#include "../../common/utils.h"

namespace py = pybind11;

class CSRMatrix {
public:
    std::vector<int> indptr;
    std::vector<int> indices;
    std::vector<double> data; // Empty if unweighted (all 1.0)
    int rows, cols;
    bool is_weighted;

    CSRMatrix(int r, int c) : rows(r), cols(c), is_weighted(false) {
        indptr.assign(r + 1, 0);
    }
};

// Power iteration with branch optimization for weighted/unweighted paths
std::vector<double> power_iteration_optimized(
    const CSRMatrix& A,
    int max_iter,
    double tol,
    std::vector<double>& x
) {
    const int n = A.rows;
    std::vector<double> x_next(n);
    bool use_weight = A.is_weighted && !A.data.empty();

    // Initial normalization
    double norm = 0.0;
    #pragma omp parallel for reduction(+:norm)
    for (int i = 0; i < n; ++i) norm += x[i] * x[i];
    norm = std::sqrt(norm);
    
    if (norm < 1e-12) {
        std::fill(x.begin(), x.end(), 1.0 / std::sqrt(n));
    } else {
        double inv_norm = 1.0 / norm;
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) x[i] *= inv_norm;
    }

    double delta = tol + 1.0;
    for (int iter = 0; iter < max_iter && delta >= tol; ++iter) {
        double next_norm_sq = 0.0;

        #pragma omp parallel for reduction(+:next_norm_sq) schedule(dynamic, 64)
        for (int i = 0; i < n; ++i) {
            double sum = 0.0;
            const int start = A.indptr[i];
            const int end = A.indptr[i+1];
            
            if (use_weight) {
                for (int j = start; j < end; ++j) {
                    sum += A.data[j] * x[A.indices[j]];
                }
            } else {
                for (int j = start; j < end; ++j) {
                    sum += x[A.indices[j]];
                }
            }
            
            x_next[i] = sum;
            next_norm_sq += sum * sum;
        }

        double next_norm = std::sqrt(next_norm_sq);
        if (next_norm < 1e-12) break;

        double inv_next_norm = 1.0 / next_norm;
        delta = 0.0;

        #pragma omp parallel for reduction(+:delta) schedule(static)
        for (int i = 0; i < n; ++i) {
            double val = x_next[i] * inv_next_norm;
            delta += std::abs(val - x[i]);
            x_next[i] = val;
        }
        x.swap(x_next);
    }
    return x;
}

// Build transpose CSR with fallback logic for missing weight keys
CSRMatrix build_transpose_matrix_smart(Graph& graph, const std::vector<node_t>& nodes, const std::string& weight_key) {
    std::shared_ptr<CSRGraph> csr_ptr = weight_key.empty() ? graph.gen_CSR() : graph.gen_CSR(weight_key);
    
    int n = static_cast<int>(nodes.size());
    CSRMatrix At(n, n);
    if (!csr_ptr) return At;

    const auto& src_indptr = csr_ptr->V;
    const auto& src_indices = csr_ptr->E;
    std::vector<double> src_data;
    bool actually_weighted = false;

    // Detect if weighted calculation is required
    if (!weight_key.empty()) {
        auto it = csr_ptr->W_map.find(weight_key);
        if (it != csr_ptr->W_map.end() && it->second) {
            src_data = *(it->second);
            for (double w : src_data) {
                if (std::abs(w - 1.0) > 1e-9) {
                    actually_weighted = true;
                    break;
                }
            }
        }
    }

    At.is_weighted = actually_weighted;

    // Calculate row counts for transpose
    for (int x_idx : src_indices) {
        if (x_idx >= 0 && x_idx < n) At.indptr[x_idx + 1]++;
    }
    for (int i = 0; i < n; ++i) At.indptr[i + 1] += At.indptr[i];

    At.indices.resize(src_indices.size());
    if (actually_weighted) At.data.resize(src_indices.size());
    
    std::vector<int> cur_pos(At.indptr.begin(), At.indptr.end());

    // Populate transpose CSR data
    for (int r = 0; r < n; ++r) {
        for (int p = src_indptr[r]; p < src_indptr[r+1]; ++p) {
            int c = src_indices[p];
            if (c < 0 || c >= n) continue;
            int dest = cur_pos[c]++;
            At.indices[dest] = r;
            if (actually_weighted) At.data[dest] = src_data[p];
        }
    }
    return At;
}

py::object cpp_eigenvector_centrality(
    py::object G,
    py::object py_max_iter,
    py::object py_tol,
    py::object py_nstart,
    py::object py_weight
) {
    try {
        Graph& graph = G.cast<Graph&>();
        int max_iter = py_max_iter.cast<int>();
        double tol = py_tol.cast<double>();
        std::string weight_key = py_weight.is_none() ? "" : py_weight.cast<std::string>();

        if (graph.node.empty()) return py::dict();

        std::vector<node_t> nodes;
        for (auto& pair : graph.node) nodes.push_back(pair.first);
        int n = nodes.size();
        
        CSRMatrix A_transpose = build_transpose_matrix_smart(graph, nodes, weight_key);
        
        // Initialize x vector (prefer degree-based or uniform)
        std::vector<double> x(n, 1.0 / n);
        if (!py_nstart.is_none()) {
            py::dict nstart = py_nstart.cast<py::dict>();
            for (int i = 0; i < n; i++) {
                py::object node_obj = graph.id_to_node[py::cast(nodes[i])];
                if (nstart.contains(node_obj)) x[i] = nstart[node_obj].cast<double>();
            }
        } else {
            for (int i = 0; i < n; i++) {
                int degree = A_transpose.indptr[i+1] - A_transpose.indptr[i];
                x[i] = (degree > 0) ? (double)degree : 1.0/n;
            }
        }

        std::vector<double> res = power_iteration_optimized(A_transpose, max_iter, tol, x);

        py::dict result;
        for (int i = 0; i < n; i++) {
            py::object node_obj = graph.id_to_node[py::cast(nodes[i])];
            result[node_obj] = res[i];
        }
        return result;

    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("C++ Eigenvector Error: ") + e.what());
    }
}