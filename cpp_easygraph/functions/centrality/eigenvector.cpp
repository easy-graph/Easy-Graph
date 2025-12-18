#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <random>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <unordered_map>
#include <tuple>
#include <omp.h> // 确保包含 OpenMP 头文件

// 假设头文件路径一致，保留原有引用
#include "centrality.h"
#include "../../classes/graph.h"

namespace py = pybind11;

// --- 移除所有 extern "C" ARPACK 声明 ---
// --- 移除所有 Eigen include ---

// CSRMatrix 稀疏矩阵结构体 (保持不变，移除 to_eigen 方法)
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

    // 并行矩阵向量乘法: result = A * vec
    void multiply_inplace(const std::vector<double>& vec, std::vector<double>& result) const {
        result.assign(rows, 0.0);
        #pragma omp parallel for schedule(static, 64) if(rows > 1000)
        for (int i = 0; i < rows; ++i) {
            double sum = 0.0;
            // 显式指示编译器进行循环展开或向量化优化
            int start = indptr[i];
            int end = indptr[i + 1];
            for (int j = start; j < end; ++j) {
                sum += data[j] * vec[indices[j]];
            }
            result[i] = sum;
        }
    }
};

// 辅助函数声明
double vector_norm(const std::vector<double>& x);
void normalize_vector(std::vector<double>& x, double norm);

// 幂迭代法 (Power Iteration)
// 这是在没有外部库时求解主特征向量的标准方法
std::vector<double> power_iteration(
    const CSRMatrix& A,
    int max_iter,
    double tol,
    std::vector<double>& x // 输入初始向量，输出结果
) {
    const int n = A.rows;
    std::vector<double> x_old(n);
    std::vector<double> x_new(n, 0.0);

    // 1. 初始化归一化
    double norm0 = vector_norm(x);
    if (norm0 < 1e-12) {
        // 如果输入向量全0，随机初始化
        std::mt19937 gen(42);
        std::uniform_real_distribution<> dis(0.0, 1.0);
        for (int i = 0; i < n; ++i) x[i] = dis(gen);
        norm0 = vector_norm(x);
    }
    normalize_vector(x, norm0);
    
    // 复制到 x_old
    std::copy(x.begin(), x.end(), x_old.begin());

    double lambda_old = 0.0;
    
    for (int iter = 0; iter < max_iter; ++iter) {
        // x_new = A * x_old
        A.multiply_inplace(x_old, x_new);

        // 计算模长
        double norm = vector_norm(x_new);
        
        // 防止下溢
        if (norm < 1e-12) break;

        // 归一化: x_new = x_new / norm
        double inv_norm = 1.0 / norm;
        #pragma omp parallel for schedule(static) if(n > 1000)
        for(int i=0; i<n; ++i) x_new[i] *= inv_norm;

        // 计算瑞利商 (Rayleigh Quotient) 近似特征值 lambda
        // lambda = (x_new^T * A * x_new) / (x_new^T * x_new)
        // 由于 x_new 已经归一化，分母为 1，且 x_new 实际上是 A*x_old 的方向
        // 这里简化判断收敛：比较 x_new 和 x_old 的差异
        
        double diff = 0.0;
        #pragma omp parallel for reduction(+:diff) if(n > 1000)
        for (int i = 0; i < n; ++i) {
            double d = x_new[i] - x_old[i]; // 注意特征向量方向可能翻转，但这在正矩阵通常不发生
            diff += d * d;
        }
        diff = std::sqrt(diff);

        // 也可以检查 lambda 的变化
        double lambda = norm; // 对于幂迭代，模长的变化率趋向于主特征值

        if (iter > 0) {
             if (diff < tol || std::abs(lambda - lambda_old) < tol * std::abs(lambda)) {
                 // 收敛
                 x = x_new;
                 return x;
             }
        }

        lambda_old = lambda;
        // 交换指针/数据，准备下一次迭代
        x_old.swap(x_new); 
    }

    x = x_old;
    return x;
}

// 辅助函数：构建转置矩阵 (逻辑保持原样，纯C++实现)
CSRMatrix build_transpose_matrix(Graph& graph, const std::vector<node_t>& nodes, const std::string& weight_key) {
    try {
        std::shared_ptr<CSRGraph> csr_ptr;
        if (weight_key.empty()) {
            csr_ptr = graph.gen_CSR();
        } else {
            csr_ptr = graph.gen_CSR(weight_key);
        }

        if (!csr_ptr) {
            CSRMatrix A(static_cast<int>(nodes.size()), static_cast<int>(nodes.size()));
            return A;
        }

        const int n = static_cast<int>(nodes.size());
        // EasyGraph 的 CSR 结构
        const auto& src_indptr = csr_ptr->V;
        const auto& src_indices = csr_ptr->E;
        std::vector<double> src_data;

        // 处理权重
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

        // --- 构建转置矩阵 At ---
        // Eigenvector Centrality 定义为 x A = lambda x，即 x 是左特征向量
        // 等价于求 A^T 的右特征向量：A^T x^T = lambda x^T
        // 所以我们需要转置矩阵
        
        int rows = n;
        int cols = n;
        CSRMatrix At(cols, rows); // 转置后行列互换，虽为方阵
        
        // 1. 计算每一列有多少个非零元素 (转置后的行度)
        for (int x : src_indices) {
            if (x >= 0 && x < cols) At.indptr[x + 1]++;
        }
        // 2. 前缀和计算 indptr
        for (int i = 0; i < cols; ++i) {
            At.indptr[i + 1] += At.indptr[i];
        }

        // 3. 填充 indices 和 data
        size_t nnz = src_indices.size();
        At.indices.resize(nnz);
        At.data.resize(nnz);
        
        std::vector<int> cur_pos(At.indptr.begin(), At.indptr.end());

        for (int r = 0; r < rows; ++r) {
            int start = src_indptr[r];
            int end = src_indptr[r+1];
            for (int p = start; p < end; ++p) {
                int c = src_indices[p]; // 原矩阵的列 -> 转置矩阵的行
                if (c < 0 || c >= cols) continue;
                
                int dest = cur_pos[c]++;
                At.indices[dest] = r; // 原矩阵的行 -> 转置矩阵的列
                At.data[dest] = (p < static_cast<int>(src_data.size())) ? src_data[p] : 1.0;
            }
        }
        return At;

    } catch (...) {
        return CSRMatrix(static_cast<int>(nodes.size()), static_cast<int>(nodes.size()));
    }
}


// 主入口函数
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

        if (graph.node.empty()) {
            return py::dict();
        }

        // 1. 映射节点 ID
        std::vector<node_t> nodes;
        nodes.reserve(graph.node.size());
        for (auto& node_pair : graph.node) {
            nodes.push_back(node_pair.first);
        }
        const int n = nodes.size();
        
        // 2. 构建转置矩阵 (A^T)
        CSRMatrix A_transpose = build_transpose_matrix(graph, nodes, weight_key);
        
        // 3. 标记孤立点 (出度为0的点在转置矩阵中表现为全0行)
        // 注意：这里的 A_transpose 行实际上代表原图的入度
        std::vector<bool> isolated_nodes(n, false);
        #pragma omp parallel for schedule(static) if(n > 1000)
        for (int i = 0; i < n; i++) {
             // 如果在转置矩阵里这一行是空的，说明原图中该点没有任何入边
             // 对于 Eigenvector Centrality，如果完全没有入边，中心性通常为0 (除非是强连通分量处理)
             if (A_transpose.indptr[i + 1] == A_transpose.indptr[i]) {
                 isolated_nodes[i] = true;
             }
        }

        // 4. 初始化向量 x
        std::vector<double> x(n, 0.0);
        
        if (py_nstart.is_none()) {
            // 默认初始化：使用度数+随机噪声，或者纯 1.0/n
            #pragma omp parallel for schedule(static) if(n > 1000)
            for (int i = 0; i < n; i++) {
                if (!isolated_nodes[i]) {
                    // 使用入度作为初始猜测通常收敛更快
                    int degree = A_transpose.indptr[i+1] - A_transpose.indptr[i];
                    x[i] = static_cast<double>(degree) + 1.0; 
                }
            }
        } else {
            // 使用用户提供的 nstart
            py::dict nstart = py_nstart.cast<py::dict>();
            for (size_t i = 0; i < nodes.size(); i++) {
                py::object node_obj = graph.id_to_node[py::cast(nodes[i])];
                if (nstart.contains(node_obj)) {
                    x[i] = nstart[node_obj].cast<double>();
                } else {
                    x[i] = 0.0; // 或者 1.0 / n
                }
            }
        }

        // 5. 执行幂迭代 (替代 ARPACK/Eigen)
        // 只需要调用这一个函数，不需要复杂的 fallback 逻辑
        std::vector<double> centrality = power_iteration(A_transpose, max_iter, tol, x);

        // 6. 后处理
        // 确保结果为正值
        double sum = 0.0;
        #pragma omp parallel for reduction(+:sum) schedule(static) if(n > 1000)
        for (int i = 0; i < n; i++) sum += centrality[i];
        
        if (sum < 0) {
             #pragma omp parallel for schedule(static) if(n > 1000)
             for(int i=0; i<n; ++i) centrality[i] = -centrality[i];
        }

        // 再次归一化 (L2 Norm)
        double norm = vector_norm(centrality);
        if (norm > 1e-12) {
            normalize_vector(centrality, norm);
        }

        // 7. 构建返回字典
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

// 辅助函数实现
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