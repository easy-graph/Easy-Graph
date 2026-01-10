#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>

#include <omp.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "centrality.h"
#include "../../classes/graph.h"

namespace py = pybind11;

class CSRMatrix {
public:
    std::vector<int> indptr;   // size rows+1
    std::vector<int> indices;  // size nnz
    std::vector<double> data;  // size nnz
    int rows = 0;
    int cols = 0;

    CSRMatrix() = default;
    CSRMatrix(int r, int c) : rows(r), cols(c) {
        indptr.assign(r + 1, 0);
    }
};

// Build transpose CSR from EasyGraph CSR so that row i contains in-neighbors of i.
static CSRMatrix build_transpose_matrix_from_csr(const std::shared_ptr<CSRGraph>& csr_ptr) {
    if (!csr_ptr) return CSRMatrix();

    const int n = static_cast<int>(csr_ptr->nodes.size());
    if (n == 0) return CSRMatrix(0, 0);

    const auto& src_indptr = csr_ptr->V;
    const auto& src_indices = csr_ptr->E;

    // Unweighted: all ones.
    std::vector<double> src_data(src_indices.size(), 1.0);

    CSRMatrix At(n, n);

    // Count nnz per column in the source (becomes nnz per row in transpose).
    for (int c : src_indices) {
        if (c >= 0 && c < n) At.indptr[c + 1]++;
    }

    // Prefix sum.
    for (int i = 0; i < n; ++i) {
        At.indptr[i + 1] += At.indptr[i];
    }

    const int nnz = static_cast<int>(src_indices.size());
    At.indices.resize(nnz);
    At.data.resize(nnz);

    std::vector<int> cur_pos(At.indptr.begin(), At.indptr.end());

    // Fill transpose.
    for (int r = 0; r < n; ++r) {
        const int start = src_indptr[r];
        const int end = src_indptr[r + 1];
        for (int p = start; p < end; ++p) {
            const int c = src_indices[p];
            if (c < 0 || c >= n) continue;
            const int dest = cur_pos[c]++;
            At.indices[dest] = r;
            At.data[dest] = src_data[p];
        }
    }

    return At;
}

static std::vector<double> katz_centrality_omp(const CSRMatrix& A,
                                               double alpha,
                                               const std::vector<double>& beta,
                                               int max_iters,
                                               double tol,
                                               bool normalize) {
    const int n = A.rows;
    std::vector<double> x(n, 1.0);      // initial guess
    std::vector<double> x_next(n, 0.0); // next iterate
    if (n == 0) return x;

    for (int iter = 0; iter < max_iters; ++iter) {
        double err_sq = 0.0;
        double norm_sq = 0.0;

        // SpMV + Katz update + error and norm in ONE pass
        #pragma omp parallel for reduction(+ : err_sq, norm_sq) schedule(static)
        for (int i = 0; i < n; ++i) {
            double sum = 0.0;
            const int row_start = A.indptr[i];
            const int row_end   = A.indptr[i + 1];

            for (int e = row_start; e < row_end; ++e) {
                sum += A.data[e] * x[A.indices[e]];
            }

            const double new_val = alpha * sum + beta[i];
            const double diff = new_val - x[i];

            x_next[i] = new_val;
            err_sq += diff * diff;
            norm_sq += new_val * new_val;
        }

        const double err  = std::sqrt(err_sq);
        const double norm = std::sqrt(norm_sq);

        x.swap(x_next);

        if (norm > 0.0 && (err / norm) < tol) {
            break;
        }
    }

    if (normalize) {
        double norm_sq2 = 0.0;
        #pragma omp parallel for reduction(+ : norm_sq2) schedule(static)
        for (int i = 0; i < n; ++i) {
            norm_sq2 += x[i] * x[i];
        }
        const double norm = std::sqrt(norm_sq2);
        if (norm > 0.0) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n; ++i) {
                x[i] /= norm;
            }
        }
    }

    return x;
}

py::object cpp_katz_centrality(py::object G,
                              py::object py_alpha,
                              py::object py_beta,
                              py::object py_max_iter,
                              py::object py_tol,
                              py::object py_normalized) {
    Graph& graph = G.cast<Graph&>();

    const double alpha = py_alpha.cast<double>();
    const int max_iter = py_max_iter.cast<int>();
    const double tol = py_tol.cast<double>();
    const bool normalized = py_normalized.cast<bool>();

    std::shared_ptr<CSRGraph> csr_ptr = graph.gen_CSR();
    if (!csr_ptr || csr_ptr->nodes.empty()) {
        return py::dict();
    }

    const int n = static_cast<int>(csr_ptr->nodes.size());

    // Build transpose CSR so that we accumulate from in-neighbors.
    CSRMatrix A = build_transpose_matrix_from_csr(csr_ptr);

    // Process beta parameter: scalar or dict(node->beta).
    std::vector<double> beta(n, 1.0);
    if (py::isinstance<py::float_>(py_beta) || py::isinstance<py::int_>(py_beta)) {
        const double beta_val = py_beta.cast<double>();
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            beta[i] = beta_val;
        }
    } else if (py::isinstance<py::dict>(py_beta)) {
        py::dict beta_dict = py_beta.cast<py::dict>();
        for (int i = 0; i < n; ++i) {
            node_t internal_id = csr_ptr->nodes[i];
            py::object node_obj = graph.id_to_node[py::cast(internal_id)];
            if (beta_dict.contains(node_obj)) {
                beta[i] = beta_dict[node_obj].cast<double>();
            }
        }
    } else {
        throw py::type_error("beta must be a float/int or a dict");
    }

    std::vector<double> scores = katz_centrality_omp(A, alpha, beta, max_iter, tol, normalized);

    // Prepare results
    py::dict result;
    for (int i = 0; i < n; ++i) {
        node_t internal_id = csr_ptr->nodes[i];
        py::object node_obj = graph.id_to_node[py::cast(internal_id)];
        result[node_obj] = scores[i];
    }

    return result;
}
