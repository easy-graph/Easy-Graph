#include <cmath>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <unordered_map>
#include <algorithm>
#include <tuple>
#include "centrality.h"
#include "../../classes/graph.h"

#ifdef EIGEN_MAJOR_VERSION
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#if EIGEN_WORLD_VERSION == 3 && EIGEN_MAJOR_VERSION >= 3
#define HAVE_EIGEN
#endif
#endif

namespace py = pybind11;

class CSRMatrix {
public:
    std::vector<int> indptr;
    std::vector<int> indices;
    std::vector<double> data;
    int rows, cols;
    
    CSRMatrix(int r, int c) : rows(r), cols(c) {
        indptr.resize(r + 1, 0);
    }
    
    void reserve(size_t nnz) {
        indices.reserve(nnz);
        data.reserve(nnz);
    }
    
    std::vector<double> multiply(const std::vector<double>& vec) const {
        std::vector<double> result(rows, 0.0);
        
        #pragma omp parallel for if(rows > 1000)
        for (int i = 0; i < rows; i++) {
            for (int j = indptr[i]; j < indptr[i + 1]; j++) {
                result[i] += data[j] * vec[indices[j]];
            }
        }
        
        return result;
    }

    void multiply_inplace(const std::vector<double>& vec, std::vector<double>& result) const {
        std::fill(result.begin(), result.end(), 0.0);
        
        #pragma omp parallel for if(rows > 1000)
        for (int i = 0; i < rows; i++) {
            for (int j = indptr[i]; j < indptr[i + 1]; j++) {
                result[i] += data[j] * vec[indices[j]];
            }
        }
    }

#ifdef HAVE_EIGEN
    Eigen::SparseMatrix<double> to_eigen() const {
        Eigen::SparseMatrix<double> eigen_mat(rows, cols);
        eigen_mat.reserve(Eigen::VectorXi::Constant(cols, 5));
        
        for (int i = 0; i < rows; i++) {
            for (int j = indptr[i]; j < indptr[i + 1]; j++) {
                eigen_mat.insert(i, indices[j]) = data[j];
            }
        }
        eigen_mat.makeCompressed();
        return eigen_mat;
    }
#endif
};

CSRMatrix build_transpose_matrix(
    const Graph& graph,
    const std::vector<node_t>& nodes,
    const std::string& weight_key
) {
    int n = nodes.size();
    std::unordered_map<node_t, int> node_to_idx;
    for (size_t i = 0; i < nodes.size(); i++) {
        node_to_idx[nodes[i]] = i;
    }
    
    size_t nnz = 0;
    for (node_t node_id : nodes) {
        const auto& adj_it = graph.adj.find(node_id);
        if (adj_it != graph.adj.end()) {
            nnz += adj_it->second.size();
        }
    }
    
    std::vector<std::tuple<int, int, double>> edges;
    edges.reserve(nnz);
    
    for (node_t src_id : nodes) {
        int src_idx = node_to_idx[src_id];
        const auto& adj_it = graph.adj.find(src_id);
        if (adj_it != graph.adj.end()) {
            for (const auto& adj_pair : adj_it->second) {
                node_t dst_id = adj_pair.first;
                auto dst_it = node_to_idx.find(dst_id);
                if (dst_it != node_to_idx.end()) {
                    int dst_idx = dst_it->second;
                    
                    double w = 1.0;
                    if (!weight_key.empty()) {
                        auto w_it = adj_pair.second.find(weight_key);
                        if (w_it != adj_pair.second.end()) {
                            w = w_it->second;
                        }
                    }
                    
                    edges.push_back(std::make_tuple(dst_idx, src_idx, w));
                }
            }
        }
    }
    
    std::sort(edges.begin(), edges.end(), 
              [](const std::tuple<int, int, double>& a, const std::tuple<int, int, double>& b) {
                  if (std::get<0>(a) != std::get<0>(b)) {
                      return std::get<0>(a) < std::get<0>(b);
                  }
                  return std::get<1>(a) < std::get<1>(b);
              });
    
    CSRMatrix matrix(n, n);
    matrix.reserve(edges.size());
    
    for (const auto& edge : edges) {
        int row = std::get<0>(edge);
        matrix.indptr[row + 1]++;
    }
    
    for (int i = 0; i < n; i++) {
        matrix.indptr[i + 1] += matrix.indptr[i];
    }
    
    for (const auto& edge : edges) {
        int row = std::get<0>(edge);
        int col = std::get<1>(edge);
        double val = std::get<2>(edge);
        
        matrix.indices.push_back(col);
        matrix.data.push_back(val);
    }
    
    std::vector<bool> has_connections(n, false);
    
    for (size_t i = 0; i < n; i++) {
        if (matrix.indptr[i + 1] > matrix.indptr[i]) {
            has_connections[i] = true;
        }
    }
    
    for (int i = 0; i < n; i++) {
        if (!has_connections[i]) {
            matrix.indices.push_back(i);
            matrix.data.push_back(1.0e-4);
            
            for (int j = i + 1; j <= n; j++) {
                matrix.indptr[j]++;
            }
        }
    }
    return matrix;
}

std::vector<double> power_iteration(
    const CSRMatrix& A,
    int max_iter,
    double tol,
    std::vector<double>& x
) {
    int n = A.rows;
    std::vector<double> x_next(n, 0.0);
    
    for (int iter = 0; iter < max_iter; iter++) {
        A.multiply_inplace(x, x_next);
        
        double norm = 0.0;
        #pragma omp parallel for reduction(+:norm) if(n > 10000)
        for (int i = 0; i < n; i++) {
            norm += x_next[i] * x_next[i];
        }
        norm = std::sqrt(norm);
        
        if (norm < 1e-12) {
            return x;
        }
        
        #pragma omp parallel for if(n > 10000)
        for (int i = 0; i < n; i++) {
            x_next[i] /= norm;
        }
        
        double diff = 0.0;
        #pragma omp parallel for reduction(+:diff) if(n > 10000)
        for (int i = 0; i < n; i++) {
            diff += std::fabs(x_next[i] - x[i]);
        }
        
        if (diff < n * tol) {
            std::swap(x, x_next);
            break;
        }
        
        std::swap(x, x_next);
    }
    
    return x;
}

#ifdef HAVE_EIGEN
std::vector<double> compute_eigenvector_eigen(
    const CSRMatrix& A, 
    int max_iter,
    double tol
) {
    Eigen::SparseMatrix<double> eigen_matrix = A.to_eigen();
    
    int n = A.rows;
    std::vector<double> result(n);
    
    try {
        Eigen::EigenSolver<Eigen::MatrixXd> solver;
        
        if (n < 1000) {
            Eigen::MatrixXd dense_matrix(eigen_matrix);
            solver.compute(dense_matrix);
            
            int max_idx = 0;
            double max_val = solver.eigenvalues()[0].real();
            for (int i = 1; i < n; i++) {
                if (solver.eigenvalues()[i].real() > max_val) {
                    max_val = solver.eigenvalues()[i].real();
                    max_idx = i;
                }
            }
            
            Eigen::VectorXd eigen_vec = solver.eigenvectors().col(max_idx).real();
            for (int i = 0; i < n; i++) {
                result[i] = eigen_vec(i);
            }
        } else {
            throw std::runtime_error("Matrix too large for dense solver");
        }
    }
    catch (const std::exception& e) {
        std::vector<double> x(n, 1.0/n);
        return power_iteration(A, max_iter, tol, x);
    }
    
    double sum = 0.0;
    for (double val : result) {
        sum += val;
    }
    
    if (sum < 0.0) {
        for (double& val : result) {
            val = -val;
        }
    }
    
    double norm = 0.0;
    for (double val : result) {
        norm += val * val;
    }
    norm = std::sqrt(norm);
    
    for (double& val : result) {
        val /= norm;
    }
    
    return result;
}
#endif

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

        if (graph.node.size() == 0) {
            return py::dict();
        }

        std::vector<node_t> nodes;
        for (auto& node_pair : graph.node) {
            nodes.push_back(node_pair.first);
        }
        int n = nodes.size();
        
        CSRMatrix A_transpose = build_transpose_matrix(graph, nodes, weight_key);
        
        std::vector<bool> isolated_nodes(n, true);
        for (int i = 0; i < n; i++) {
            // A node is isolated if it has zero edges (no entries in its row)
            if (A_transpose.indptr[i + 1] == A_transpose.indptr[i]) {
                isolated_nodes[i] = true;
            } else {
                isolated_nodes[i] = false;
            }
        }

        std::vector<double> centrality;
        
        if (py_nstart.is_none()) {
            bool fast_solver_success = false;
            
#ifdef HAVE_EIGEN
            try {
                centrality = compute_eigenvector_eigen(A_transpose, max_iter, tol);
                fast_solver_success = true;
            } catch (const std::exception&) {
                fast_solver_success = false;
            }
#endif
            
            if (!fast_solver_success) {
                std::vector<double> x(n, 1.0/n);
                centrality = power_iteration(A_transpose, max_iter, tol, x);
            }
        } else {
            py::dict nstart = py_nstart.cast<py::dict>();
            std::vector<double> x(n, 0.0);
            
            for (size_t i = 0; i < nodes.size(); i++) {
                py::object node_obj = graph.id_to_node[py::cast(nodes[i])];
                if (nstart.contains(node_obj)) {
                    x[i] = nstart[node_obj].cast<double>();
                } else {
                    x[i] = 1.0;
                }
            }
            
            bool all_zeros = true;
            for (double val : x) {
                if (std::abs(val) > 1e-10) {
                    all_zeros = false;
                    break;
                }
            }
            
            if (all_zeros) {
                throw std::runtime_error("initial vector cannot have all zero values");
            }
            
            double sum_abs = 0.0;
            for (double val : x) {
                sum_abs += std::fabs(val);
            }
            
            for (double& val : x) {
                val /= sum_abs;
            }
            
            centrality = power_iteration(A_transpose, max_iter, tol, x);
        }
        
        double sum = 0.0;
        for (double val : centrality) {
            sum += val;
        }
        
        if (sum < 0.0) {
            for (double& val : centrality) {
                val = -val;
            }
        }
        
        for (int i = 0; i < n; i++) {
            if (isolated_nodes[i]) {
                centrality[i] = 0.0;
            }
        }

        double norm = 0.0;
        for (double val : centrality) {
            norm += val * val;
        }
        norm = std::sqrt(norm);
        
        if (norm > 0.0) {
            for (double& val : centrality) {
                val /= norm;
            }
        }
        
        py::dict result;
        for (size_t i = 0; i < nodes.size(); i++) {
            py::object node_obj = graph.id_to_node[py::cast(nodes[i])];
            result[node_obj] = centrality[i];
        }
        
        return result;
    } catch (const std::exception& e) {
        throw std::runtime_error(e.what());
    }
}