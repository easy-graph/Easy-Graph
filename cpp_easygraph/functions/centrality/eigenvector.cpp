#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <random>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <omp.h>

#include "centrality.h"
#include "../../classes/graph.h"

namespace py = pybind11;

class CSRMatrix {
public:
    std::vector<int> indptr;
    std::vector<int> indices;
    std::vector<double> data;
    int rows;
    int cols;

    CSRMatrix() : rows(0), cols(0) {}
    CSRMatrix(int r, int c) : rows(r), cols(c) {
        indptr.assign(r + 1, 0);
    }
};

std::vector<double> power_iteration_optimized(
    const CSRMatrix& A,
    int max_iter,
    double tol,
    std::vector<double>& x
) {
    const int n = A.rows;
    std::vector<double> x_next(n);
    
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
            
            for (int j = start; j < end; ++j) {
                sum += A.data[j] * x[A.indices[j]];
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

CSRMatrix build_transpose_matrix(Graph& graph, const std::vector<node_t>& nodes, const std::string& weight_key) {
    try {
        std::shared_ptr<CSRGraph> csr_ptr;
        if (weight_key.empty()) {
            csr_ptr = graph.gen_CSR();
        } else {
            csr_ptr = graph.gen_CSR(weight_key);
        }

        if (!csr_ptr) return CSRMatrix(nodes.size(), nodes.size());

        const int n = static_cast<int>(nodes.size());
        const auto& src_indptr = csr_ptr->V;
        const auto& src_indices = csr_ptr->E;
        std::vector<double> src_data;

        if (weight_key.empty()) {
             src_data = csr_ptr->unweighted_W.empty() ? 
                       std::vector<double>(csr_ptr->E.size(), 1.0) : 
                       csr_ptr->unweighted_W;
        } else {
            auto it = csr_ptr->W_map.find(weight_key);
            if (it != csr_ptr->W_map.end() && it->second) {
                src_data = *(it->second);
            } else {
                src_data = std::vector<double>(csr_ptr->E.size(), 1.0);
            }
        }

        int rows = n;
        int cols = n;
        CSRMatrix At(cols, rows);
        
        for (int x : src_indices) {
            if (x >= 0 && x < cols) At.indptr[x + 1]++;
        }
        for (int i = 0; i < cols; ++i) {
            At.indptr[i + 1] += At.indptr[i];
        }

        size_t nnz = src_indices.size();
        At.indices.resize(nnz);
        At.data.resize(nnz);
        std::vector<int> cur_pos(At.indptr.begin(), At.indptr.end());

        for (int r = 0; r < rows; ++r) {
            int start = src_indptr[r];
            int end = src_indptr[r+1];
            for (int p = start; p < end; ++p) {
                int c = src_indices[p];
                if (c < 0 || c >= cols) continue;
                int dest = cur_pos[c]++;
                At.indices[dest] = r;
                At.data[dest] = (p < static_cast<int>(src_data.size())) ? src_data[p] : 1.0;
            }
        }
        return At;
    } catch (...) {
        return CSRMatrix(nodes.size(), nodes.size());
    }
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
        std::string weight_key = "";
        if (!py_weight.is_none()) {
            weight_key = py_weight.cast<std::string>();
        }

        if (graph.node.empty()) return py::dict();

        std::vector<node_t> nodes;
        nodes.reserve(graph.node.size());
        for (auto& node_pair : graph.node) {
            nodes.push_back(node_pair.first);
        }
        const int n = nodes.size();
        
        CSRMatrix A_transpose = build_transpose_matrix(graph, nodes, weight_key);
        
        std::vector<double> x(n, 0.0);
        
        if (py_nstart.is_none()) {
            #pragma omp parallel for
            for (int i = 0; i < n; i++) {
                if (A_transpose.indptr[i + 1] != A_transpose.indptr[i]) {
                    x[i] = static_cast<double>(A_transpose.indptr[i+1] - A_transpose.indptr[i]); 
                } else {
                    x[i] = 1.0 / n;
                }
            }
        } else {
            py::dict nstart = py_nstart.cast<py::dict>();
            for (size_t i = 0; i < nodes.size(); i++) {
                py::object node_obj = graph.id_to_node[py::cast(nodes[i])];
                if (nstart.contains(node_obj)) {
                    x[i] = nstart[node_obj].cast<double>();
                } else {
                    x[i] = 0.0;
                }
            }
        }

        std::vector<double> centrality = power_iteration_optimized(A_transpose, max_iter, tol, x);

        py::dict result;
        for (size_t i = 0; i < nodes.size(); i++) {
            py::object node_obj = graph.id_to_node[py::cast(nodes[i])];
            result[node_obj] = centrality[i];
        }
        return result;

    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("C++ exception: ") + e.what());
    }
}