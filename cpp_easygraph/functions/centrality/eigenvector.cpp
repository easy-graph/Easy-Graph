#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <unordered_map>
#include <tuple>
#include "centrality.h"
#include "../../classes/graph.h"

#ifdef HAVE_EIGEN
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
    #ifdef HAVE_EIGEN_SPARSE_SOLVER
    #include <Eigen/SparseCore>
    #endif
#endif

namespace py = pybind11;

extern "C" {
    void dsaupd_(int*, char*, int*, char*, int*, double*, double*, int*, 
                 double*, int*, int*, int*, double*, double*, int*, int*);
    void dseupd_(int*, char*, int*, double*, double*, int*, double*, 
                 char*, int*, char*, int*, double*, double*, int*, 
                 double*, int*, int*, int*, double*, double*, int*, int*);
    void dnaupd_(int*, char*, int*, char*, int*, double*, double*, int*, 
                 double*, int*, int*, int*, double*, double*, int*, int*);
    void dneupd_(int* rvec, char* howmny, int* select,
                 double* dr, double* di, double* z, int* ldz,
                 double* sigmar, double* sigmai, double* workev,
                 char* bmat, int* n, char* which, int* nev,
                 double* tol, double* resid, int* ncv, double* v, int* ldv,
                 int* iparam, int* ipntr, double* workd, double* workl,
                 int* lworkl, int* info);
}

// CSRMatrix 稀疏矩阵结构体
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

    void reserve(size_t nnz) {
        indices.reserve(nnz);
        data.reserve(nnz);
    }

    void multiply_inplace(const std::vector<double>& vec, std::vector<double>& result) const {
        result.assign(rows, 0.0);
        #pragma omp parallel for schedule(static,64) if(rows > 1000)
        for (int i = 0; i < rows; ++i) {
            double sum = 0.0;
            for (int j = indptr[i]; j < indptr[i + 1]; ++j) {
                sum += data[j] * vec[indices[j]];
            }
            result[i] = sum;
        }
    }

    double estimate_norm() const {
        double s = 0.0;
        #pragma omp parallel for reduction(+:s) if(data.size() > 1000)
        for (size_t i = 0; i < data.size(); ++i) s += data[i] * data[i];
        return std::sqrt(s);
    }

    double rayleigh_quotient(const std::vector<double>& x, const std::vector<double>& Ax) const {
        double num = 0.0, den = 0.0;
        #pragma omp parallel for reduction(+:num,den) if(rows > 1000)
        for (int i = 0; i < rows; ++i) {
            num += Ax[i] * x[i];
            den += x[i] * x[i];
        }
        return den > 0.0 ? num / den : 0.0;
    }

#ifdef HAVE_EIGEN
    Eigen::SparseMatrix<double> to_eigen() const {
        Eigen::SparseMatrix<double> M(rows, cols);
        std::vector<Eigen::Triplet<double>> trip;
        trip.reserve(data.size());
        for (int i = 0; i < rows; ++i) {
            for (int j = indptr[i]; j < indptr[i+1]; ++j) {
                trip.emplace_back(i, indices[j], data[j]);
            }
        }
        M.setFromTriplets(trip.begin(), trip.end());
        M.makeCompressed();
        return M;
    }
#endif
};

// 前向声明
std::vector<double> power_iteration_chebyshev(
    const CSRMatrix& A,
    int max_iter,
    double tol,
    std::vector<double>& x
);
double vector_norm(const std::vector<double>& x);
void normalize_vector(std::vector<double>& x, double norm);
double vector_diff_norm(const std::vector<double>& x1, const std::vector<double>& x2);



// 构建转置稀疏矩阵
CSRMatrix build_transpose_matrix(Graph& graph, const std::vector<node_t>& nodes, const std::string& weight_key) {
    try {
        std::shared_ptr<CSRGraph> csr_ptr;
        if (weight_key.empty()) {
            csr_ptr = graph.gen_CSR();
        } else {
            csr_ptr = graph.gen_CSR(weight_key);
        }

        if (!csr_ptr) {
            std::cerr << "[build_transpose_matrix] Error: gen_CSR returned nullptr" << std::endl;
            const int n = static_cast<int>(nodes.size());
            CSRMatrix A(n, n);
            A.indptr.assign(n + 1, 0);
            return A;
        }

        const int n = static_cast<int>(nodes.size());
        std::vector<int> src_indptr;
        std::vector<int> src_indices;
        std::vector<double> src_data;
        int rows = 0, cols = 0;

        // 只用V/E/unweighted_W/W_map
        if (!csr_ptr->V.empty() && !csr_ptr->E.empty()) {
            rows = static_cast<int>(csr_ptr->V.size()) - 1;
            cols = rows;
            src_indptr = csr_ptr->V;
            src_indices = csr_ptr->E;
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
        } else {
            std::cerr << "[build_transpose_matrix] Warning: CSR data is empty (V size=" 
                      << csr_ptr->V.size() << ")" << std::endl;
            CSRMatrix A(n, n);
            A.indptr.assign(n + 1, 0);
            return A;
        }

        if (src_indptr.size() < 2 || src_indices.empty()) {
            std::cerr << "[build_transpose_matrix] Warning: Invalid CSR structure" << std::endl;
            CSRMatrix A(n, n);
            A.indptr.assign(n + 1, 0);
            return A;
        }

        // 转置
        CSRMatrix At(cols, rows);
        At.indptr.assign(cols + 1, 0);
        size_t nnz = src_indices.size();

        for (size_t idx = 0; idx < nnz; ++idx) {
            int col = src_indices[idx];
            if (col >= 0 && col < cols) {
                At.indptr[col + 1]++;
            }
        }
        for (int i = 0; i < cols; ++i) {
            At.indptr[i + 1] += At.indptr[i];
        }

        At.indices.assign(nnz, 0);
        At.data.assign(nnz, 0.0);

        std::vector<int> cur_pos(At.indptr.begin(), At.indptr.end());
        
        for (int r = 0; r < rows; ++r) {
            int start = src_indptr[r];
            int end = (r + 1 < static_cast<int>(src_indptr.size())) ? src_indptr[r + 1] : static_cast<int>(nnz);
            for (int p = start; p < end; ++p) {
                if (p >= static_cast<int>(src_indices.size())) break;
                int c = src_indices[p];
                if (c < 0 || c >= cols) continue;
                int dest = cur_pos[c]++;
                At.indices[dest] = r;
                At.data[dest] = (p < static_cast<int>(src_data.size())) ? src_data[p] : 1.0;
            }
        }

        return At;
    } catch (const std::exception& e) {
        std::cerr << "[build_transpose_matrix] Exception: " << e.what() << std::endl;
        const int n = static_cast<int>(nodes.size());
        CSRMatrix A(n, n);
        A.indptr.assign(n + 1, 0);
        return A;
    }
}

// ARPACK: 有向图
std::vector<double> compute_eigenvector_arpack_directed(
    const CSRMatrix& A,
    int max_iter,
    double tol
) {
    const int n_const = A.rows;
    std::vector<double> result(n_const, 0.0);
    int n = n_const;
    int nev = 1;
    int ncv = (n > 30) ? 30 : n;
    if (ncv < nev + 2) {
        ncv = std::min(nev + 2, n);
    }
    int ido = 0;
    char bmat[2] = "I";
    char which[3] = "LR";
    std::vector<double> resid(n);
    std::vector<double> v(static_cast<size_t>(n) * ncv);
    std::vector<double> workd(3 * n);
    int lworkl = 3 * ncv * ncv + 6 * ncv;
    std::vector<double> workl(lworkl);

    #pragma omp parallel for schedule(static) if(n > 1000)
    for (int i = 0; i < n; i++) {
        int degree = A.indptr[i + 1] - A.indptr[i];
        resid[i] = degree > 0 ? 
                   static_cast<double>(degree) + 1e-4 * (rand() / static_cast<double>(RAND_MAX)) :
                   1.0;
    }
    double norm = 0.0;
    for (int i = 0; i < n; i++) {
        norm += resid[i] * resid[i];
    }
    norm = std::sqrt(norm);
    if (norm > 1e-12) {
        for (int i = 0; i < n; i++) {
            resid[i] /= norm;
        }
    }
    int iparam[11] = {0};
    iparam[0] = 1;
    iparam[2] = max_iter;
    iparam[3] = 1;
    iparam[6] = 1;
    int ipntr[14];
    int ldv = n;
    int info = 0;
    int iter_count = 0;
    auto start_time = std::chrono::high_resolution_clock::now();
    try {
        while (true) {
            dnaupd_(&ido, bmat, &n, which, &nev, &tol, resid.data(), &ncv,
                    v.data(), &ldv, iparam, ipntr, workd.data(), workl.data(),
                    &lworkl, &info);
            if (ido == -1 || ido == 1) {
                int x_idx = ipntr[0] - 1;
                int y_idx = ipntr[1] - 1;
                std::vector<double> x_vec(n), y_vec(n);
                std::copy(workd.begin() + x_idx, workd.begin() + x_idx + n, x_vec.begin());
                A.multiply_inplace(x_vec, y_vec);
                std::copy(y_vec.begin(), y_vec.end(), workd.begin() + y_idx);
                iter_count++;
            } else if (ido == 99) {
                break;
            } else {
                throw std::runtime_error("ARPACK unexpected ido: " + std::to_string(ido));
            }
            if (iter_count > max_iter * 10) {
                throw std::runtime_error("ARPACK exceeded iteration limit");
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[ARPACK-Directed] Error: " << e.what() << std::endl;
        throw;
    }
    if (info < 0) {
        std::string error_msg = "ARPACK dnaupd_ error: " + std::to_string(info);
        switch(info) {
            case -5: error_msg += " (LWORKL too small, need at least " + std::to_string(3*ncv*ncv + 6*ncv) + ")"; break;
        }
        throw std::runtime_error(error_msg);
    }
    int rvec = 1;
    char howmny[2] = "A";
    std::vector<int> select(ncv);
    std::vector<double> dr(nev + 1);
    std::vector<double> di(nev + 1);
    std::vector<double> z(n * (nev + 1));
    std::vector<double> workev(3 * ncv);
    int ldz = n;
    double sigmar = 0.0, sigmai = 0.0;
    dneupd_(&rvec, howmny, select.data(), dr.data(), di.data(), z.data(), &ldz,
            &sigmar, &sigmai, workev.data(), bmat, &n, which, &nev, &tol,
            resid.data(), &ncv, v.data(), &ldv, iparam, ipntr, workd.data(),
            workl.data(), &lworkl, &info);
    if (info != 0) {
        throw std::runtime_error("ARPACK dneupd_ error: " + std::to_string(info));
    }
    std::copy(z.begin(), z.begin() + n, result.begin());
    return result;
}

// ARPACK: 无向图
std::vector<double> compute_eigenvector_arpack(
    const CSRMatrix& A,
    int max_iter,
    double tol
) {
    const int n_const = A.rows;
    std::vector<double> result(n_const, 0.0);
    int n = n_const;
    int nev = 1;
    int ncv = (n > 30) ? 30 : n;
    if (ncv < nev + 2) {
        ncv = std::min(nev + 2, n);
    }
    int ido = 0;
    char bmat[2] = "I";
    char which[3] = "LR";
    std::vector<double> resid(n);
    std::vector<double> v(static_cast<size_t>(n) * ncv);
    std::vector<double> workd(3 * n);
    int lworkl = ncv * (ncv + 8);
    std::vector<double> workl(lworkl);

    #pragma omp parallel for schedule(static) if(n > 1000)
    for (int i = 0; i < n; i++) {
        int degree = A.indptr[i + 1] - A.indptr[i];
        resid[i] = degree > 0 ? 
                   static_cast<double>(degree) + 1e-4 * (rand() / static_cast<double>(RAND_MAX)) :
                   1.0;
    }
    double norm = 0.0;
    for (int i = 0; i < n; i++) {
        norm += resid[i] * resid[i];
    }
    norm = std::sqrt(norm);
    if (norm > 1e-12) {
        for (int i = 0; i < n; i++) {
            resid[i] /= norm;
        }
    }
    int iparam[11] = {0};
    iparam[0] = 1;
    iparam[2] = max_iter;
    iparam[3] = 1;
    iparam[6] = 1;
    int ipntr[11];
    int ldv = n;
    int info = 0;
    int iter_count = 0;
    auto start_time = std::chrono::high_resolution_clock::now();
    try {
        while (true) {
            dsaupd_(&ido, bmat, &n, which, &nev, &tol, resid.data(), &ncv,
                    v.data(), &ldv, iparam, ipntr, workd.data(), workl.data(),
                    &lworkl, &info);
            if (ido == -1 || ido == 1) {
                int x_idx = ipntr[0] - 1;
                int y_idx = ipntr[1] - 1;
                std::vector<double> x_vec(n), y_vec(n);
                std::copy(workd.begin() + x_idx, workd.begin() + x_idx + n, x_vec.begin());
                A.multiply_inplace(x_vec, y_vec);
                std::copy(y_vec.begin(), y_vec.end(), workd.begin() + y_idx);
                iter_count++;
            } else if (ido == 99) {
                break;
            } else {
                throw std::runtime_error("ARPACK unexpected ido: " + std::to_string(ido));
            }
            if (iter_count > max_iter * 10) {
                throw std::runtime_error("ARPACK exceeded iteration limit");
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[ARPACK] Error: " << e.what() << std::endl;
        throw;
    }
    if (info < 0) {
        std::string error_msg = "ARPACK dsaupd_ error: " + std::to_string(info);
        switch(info) {
            case -1: error_msg += " (N must be positive)"; break;
            case -2: error_msg += " (NEV must be positive)"; break;
            case -3: error_msg += " (NCV-NEV >= 2 and NCV <= N not satisfied)"; break;
            case -5: error_msg += " (LWORKL too small, need at least " + std::to_string(ncv * (ncv + 8)) + ")"; break;
            case -8: error_msg += " (Error in LAPACK)"; break;
            case -9: error_msg += " (Starting vector is zero)"; break;
            case -14: error_msg += " (Did not find any eigenvalues)"; break;
        }
        throw std::runtime_error(error_msg);
    }
    int rvec = 1;
    char howmny[2] = "A";
    std::vector<int> select(ncv);
    std::vector<double> d(nev);
    std::vector<double> z(n * nev);
    int ldz = n;
    double sigma = 0.0;
    dseupd_(&rvec, howmny, select.data(), d.data(), z.data(), &ldz, &sigma,
            bmat, &n, which, &nev, &tol, resid.data(), &ncv, v.data(), &ldv,
            iparam, ipntr, workd.data(), workl.data(), &lworkl, &info);
    if (info != 0) {
        throw std::runtime_error("ARPACK dseupd_ error: " + std::to_string(info));
    }
    std::copy(z.begin(), z.begin() + n, result.begin());
    return result;
}

#if defined(HAVE_EIGEN) && defined(HAVE_EIGEN_SPARSE_SOLVER)
std::vector<double> compute_eigenvector_eigen_sparse(
    const CSRMatrix& A,
    int max_iter,
    double tol
) {
    const int n = A.rows;
    Eigen::SparseMatrix<double> eigen_mat = A.to_eigen();
    Eigen::VectorXd x(n);
    #pragma omp parallel for schedule(static) if(n > 1000)
    for (int i = 0; i < n; i++) {
        int degree = A.indptr[i + 1] - A.indptr[i];
        double second_order = 0.0;
        for (int j = A.indptr[i]; j < A.indptr[i + 1]; j++) {
            int nei = A.indices[j];
            int nei_degree = A.indptr[nei + 1] - A.indptr[nei];
            second_order += nei_degree;
        }
        x(i) = degree > 0 ? 
               static_cast<double>(degree) + 0.1 * second_order / (degree + 1) :
               1.0;
    }
    x.normalize();
    double prev_eigenvalue = 0.0;
    const double rel_tol = tol * std::max(1.0, A.estimate_norm());
    for (int iter = 0; iter < max_iter; iter++) {
        Eigen::VectorXd x_new = eigen_mat * x;
        double norm = x_new.norm();
        if (norm < 1e-12) break;
        x_new /= norm;
        double eigenvalue = x_new.dot(eigen_mat * x_new) / x_new.squaredNorm();
        double diff = (x_new - x).norm();
        bool converged_vector = (diff < rel_tol);
        bool converged_eigenvalue = (iter > 0) && 
                                   (std::abs(eigenvalue - prev_eigenvalue) < tol * std::abs(eigenvalue));
        if (converged_vector || converged_eigenvalue) {
            x = x_new;
            break;
        }
        x = x_new;
        prev_eigenvalue = eigenvalue;
    }
    std::vector<double> result(n);
    #pragma omp parallel for schedule(static) if(n > 1000)
    for (int i = 0; i < n; i++) {
        result[i] = x(i);
    }
    return result;
}
#endif

// 主入口
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
        nodes.reserve(graph.node.size());
        for (auto& node_pair : graph.node) {
            nodes.push_back(node_pair.first);
        }
        const int n = nodes.size();
        
        CSRMatrix A_transpose = build_transpose_matrix(graph, nodes, weight_key);
        
        std::vector<bool> isolated_nodes(n, false);
        #pragma omp parallel for schedule(static) if(n > 1000)
        for (int i = 0; i < n; i++) {
            isolated_nodes[i] = (A_transpose.indptr[i + 1] == A_transpose.indptr[i]);
        }

        std::vector<double> centrality;
        
        if (py_nstart.is_none()) {
            bool fast_solver_success = false;
#ifdef HAVE_ARPACK
            try {
                if (Graph_is_directed(G)) {
                    centrality = compute_eigenvector_arpack_directed(A_transpose, max_iter, tol);
                } else {
                    centrality = compute_eigenvector_arpack(A_transpose, max_iter, tol);
                }
                fast_solver_success = true;
            } catch (const std::exception& e) {
                fast_solver_success = false;
            }
#endif
#if defined(HAVE_EIGEN) && defined(HAVE_EIGEN_SPARSE_SOLVER)
            if (!fast_solver_success && n >= 100 && n < 500000) {
                try {
                    centrality = compute_eigenvector_eigen_sparse(A_transpose, max_iter, tol);
                    fast_solver_success = true;
                } catch (const std::exception&) {
                    fast_solver_success = false;
                }
            }
#endif
            if (!fast_solver_success) {
                std::vector<double> x(n, 0.0);
                #pragma omp parallel for schedule(static) if(n > 1000)
                for (int i = 0; i < n; i++) {
                    if (!isolated_nodes[i]) {
                        const int degree = A_transpose.indptr[i + 1] - A_transpose.indptr[i];
                        double second_order = 0.0;
                        for (int j = A_transpose.indptr[i]; j < A_transpose.indptr[i + 1]; j++) {
                            int nei = A_transpose.indices[j];
                            int nei_degree = A_transpose.indptr[nei + 1] - A_transpose.indptr[nei];
                            second_order += nei_degree;
                        }
                        x[i] = static_cast<double>(degree) + 
                               0.1 * second_order / (degree + 1) +
                               1e-4 * (rand() / static_cast<double>(RAND_MAX));
                    }
                }
                centrality = power_iteration_chebyshev(A_transpose, max_iter, tol, x);
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
            #pragma omp parallel for reduction(+:sum_abs) schedule(static) if(n > 1000)
            for (int i = 0; i < n; i++) {
                sum_abs += std::fabs(x[i]);
            }
            const double inv_sum = 1.0 / sum_abs;
            #pragma omp parallel for schedule(static) if(n > 1000)
            for (int i = 0; i < n; i++) {
                x[i] *= inv_sum;
            }
            centrality = power_iteration_chebyshev(A_transpose, max_iter, tol, x);
        }
        double sum = 0.0;
        #pragma omp parallel for reduction(+:sum) schedule(static) if(n > 1000)
        for (int i = 0; i < n; i++) {
            sum += centrality[i];
        }
        if (sum < 0.0) {
            #pragma omp parallel for schedule(static) if(n > 1000)
            for (int i = 0; i < n; i++) {
                centrality[i] = -centrality[i];
            }
        }
        #pragma omp parallel for schedule(static) if(n > 1000)
        for (int i = 0; i < n; i++) {
            if (isolated_nodes[i]) {
                centrality[i] = 0.0;
            }
        }
        double norm = vector_norm(centrality);
        if (norm > 1e-12) {
            normalize_vector(centrality, norm);
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

// 幂迭代
std::vector<double> power_iteration_chebyshev(
    const CSRMatrix& A,
    int max_iter,
    double tol,
    std::vector<double>& x
) {
    const int n = A.rows;
    std::vector<double> x_old(n, 0.0);
    std::vector<double> x_new(n, 0.0);
    double norm0 = 0.0;
    for (int i = 0; i < n; ++i) norm0 += x[i] * x[i];
    norm0 = std::sqrt(norm0);
    if (norm0 < 1e-12) {
        for (int i = 0; i < n; ++i) x_old[i] = 1.0 / std::sqrt(static_cast<double>(n));
    } else {
        for (int i = 0; i < n; ++i) x_old[i] = x[i] / norm0;
    }
    double lambda_old = 0.0, lambda = 0.0;
    for (int iter = 0; iter < max_iter; ++iter) {
        A.multiply_inplace(x_old, x_new);
        double norm = 0.0;
        for (int i = 0; i < n; ++i) norm += x_new[i] * x_new[i];
        norm = std::sqrt(norm);
        if (norm < 1e-12) break;
        double inv_norm = 1.0 / norm;
        for (int i = 0; i < n; ++i) x_new[i] *= inv_norm;
        std::vector<double> Ax(n, 0.0);
        A.multiply_inplace(x_new, Ax);
        double num = 0.0, den = 0.0;
        for (int i = 0; i < n; ++i) {
            num += x_new[i] * Ax[i];
            den += x_new[i] * x_new[i];
        }
        lambda = (den > 0.0) ? (num / den) : 0.0;
        double diff = 0.0;
        for (int i = 0; i < n; ++i) {
            double d = x_new[i] - x_old[i];
            diff += d * d;
        }
        diff = std::sqrt(diff);
        if ((iter > 0 && std::abs(lambda - lambda_old) < tol * std::abs(lambda)) || (diff < tol)) {
            for (int i = 0; i < n; ++i) x[i] = x_new[i];
            return x;
        }
        lambda_old = lambda;
        x_old.swap(x_new);
    }
    for (int i = 0; i < n; ++i) x[i] = x_old[i];
    return x;
}

double vector_norm(const std::vector<double>& x) {
    double s = 0.0;
    #pragma omp parallel for reduction(+:s) if(x.size() > 1000)
    for (size_t i = 0; i < x.size(); ++i) s += x[i] * x[i];
    return std::sqrt(s);
}

void normalize_vector(std::vector<double>& x, double norm) {
    const double inv_norm = 1.0 / norm;
    #pragma omp parallel for schedule(static) if(x.size() > 1000)
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] *= inv_norm;
    }
}

double vector_diff_norm(const std::vector<double>& x1, const std::vector<double>& x2) {
    double diff = 0.0;
    #pragma omp parallel for reduction(+:diff) if(x1.size() > 1000)
    for (size_t i = 0; i < x1.size(); ++i) {
        double d = x1[i] - x2[i];
        diff += d * d;
    }
    return std::sqrt(diff);
}