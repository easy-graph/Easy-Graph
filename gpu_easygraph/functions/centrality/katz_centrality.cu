#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <vector>

#include "../../gpu_easygraph.h"

namespace gpu_easygraph {
namespace {

constexpr int kBlockSize = 256;

// This implementation follows the convergence semantics of RAPIDS cuGraph's
// Katz core (Apache-2.0): double-buffered power iteration, GPU L1 residual,
// and failure when the iteration limit is reached. It uses EGGPU's compact
// incoming-CSR interface instead of cuGraph's RAFT/RMM graph infrastructure.
// Source consulted: https://github.com/rapidsai/cugraph/blob/main/cpp/src/centrality/katz_centrality_impl.cuh
__global__ void katz_iteration_kernel(const int* offsets,
                                      const int* columns,
                                      const double* current,
                                      double* next,
                                      double* absolute_deltas,
                                      int nodes,
                                      double alpha,
                                      double beta) {
    const int vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex < nodes) {
        double sum = 0.0;
        for (int edge = offsets[vertex]; edge < offsets[vertex + 1]; ++edge) {
            sum += current[columns[edge]];
        }
        const double value = alpha * sum + beta;
        next[vertex] = value;
        absolute_deltas[vertex] = fabs(value - current[vertex]);
    }
}

__global__ void square_scores_kernel(const double* scores, double* squares, int nodes) {
    const int vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex < nodes) squares[vertex] = scores[vertex] * scores[vertex];
}

__global__ void normalize_kernel(double* scores, int nodes, double norm) {
    const int vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex < nodes) scores[vertex] /= norm;
}

bool cuda_ok(cudaError_t status) {
    return status == cudaSuccess;
}

KatzResult make_result(KatzStatus status, int iterations, double residual) {
    return KatzResult{status, iterations, residual};
}

}  // namespace

std::unique_ptr<PreparedKatzContext> PreparedKatzContext::create(
    const std::vector<int>& incoming_offsets,
    const std::vector<int>& incoming_columns,
    KatzStatus& status) {
    status = KatzStatus::invalid_input;
    if (incoming_offsets.size() < 2 || incoming_offsets.front() != 0 ||
        incoming_offsets.back() != static_cast<int>(incoming_columns.size())) {
        return nullptr;
    }

    std::unique_ptr<PreparedKatzContext> context(new PreparedKatzContext());
    context->nodes_ = static_cast<int>(incoming_offsets.size()) - 1;
    for (int vertex = 0; vertex < context->nodes_; ++vertex) {
        const int begin = incoming_offsets[vertex];
        const int end = incoming_offsets[vertex + 1];
        if (begin < 0 || end < begin) return nullptr;
        context->max_in_degree_ = std::max(context->max_in_degree_, end - begin);
    }
    for (int neighbor : incoming_columns) {
        if (neighbor < 0 || neighbor >= context->nodes_) return nullptr;
    }

    if (!cuda_ok(cudaMalloc(&context->d_offsets_, incoming_offsets.size() * sizeof(int))) ||
        (!incoming_columns.empty() &&
         !cuda_ok(cudaMalloc(&context->d_columns_, incoming_columns.size() * sizeof(int)))) ||
        !cuda_ok(cudaMalloc(&context->d_current_, static_cast<size_t>(context->nodes_) * sizeof(double))) ||
        !cuda_ok(cudaMalloc(&context->d_next_, static_cast<size_t>(context->nodes_) * sizeof(double))) ||
        !cuda_ok(cudaMalloc(&context->d_values_, static_cast<size_t>(context->nodes_) * sizeof(double))) ||
        !cuda_ok(cudaMalloc(&context->d_reduce_result_, sizeof(double))) ||
        !cuda_ok(cudaMemcpy(context->d_offsets_, incoming_offsets.data(),
                            incoming_offsets.size() * sizeof(int), cudaMemcpyHostToDevice)) ||
        (!incoming_columns.empty() &&
         !cuda_ok(cudaMemcpy(context->d_columns_, incoming_columns.data(),
                             incoming_columns.size() * sizeof(int), cudaMemcpyHostToDevice))) ||
        !cuda_ok(cub::DeviceReduce::Sum(nullptr, context->reduce_temp_bytes_, context->d_values_,
                                        context->d_reduce_result_, context->nodes_)) ||
        !cuda_ok(cudaMalloc(&context->d_reduce_temp_, context->reduce_temp_bytes_))) {
        status = KatzStatus::cuda_error;
        return nullptr;
    }

    status = KatzStatus::success;
    return context;
}

PreparedKatzContext::~PreparedKatzContext() {
    if (d_reduce_temp_) cudaFree(d_reduce_temp_);
    if (d_reduce_result_) cudaFree(d_reduce_result_);
    if (d_values_) cudaFree(d_values_);
    if (d_next_) cudaFree(d_next_);
    if (d_current_) cudaFree(d_current_);
    if (d_columns_) cudaFree(d_columns_);
    if (d_offsets_) cudaFree(d_offsets_);
}

KatzResult PreparedKatzContext::run(double alpha,
                                    double beta,
                                    int max_iter,
                                    double epsilon,
                                    bool normalized,
                                    std::vector<double>& scores) {
    scores.clear();
    if (max_iter < 1 || epsilon <= 0.0 || !std::isfinite(alpha) || !std::isfinite(beta) ||
        alpha < 0.0 ||
        (max_in_degree_ > 0 && !(alpha < 1.0 / static_cast<double>(max_in_degree_)))) {
        return make_result(KatzStatus::invalid_input, 0, 0.0);
    }

    const int blocks = (nodes_ + kBlockSize - 1) / kBlockSize;
    if (!cuda_ok(cudaMemset(d_current_, 0, static_cast<size_t>(nodes_) * sizeof(double)))) {
        return make_result(KatzStatus::cuda_error, 0, 0.0);
    }

    int iterations = 0;
    double residual = 0.0;
    bool converged = false;
    for (int iteration = 0; iteration < max_iter; ++iteration) {
        katz_iteration_kernel<<<blocks, kBlockSize>>>(
            d_offsets_, d_columns_, d_current_, d_next_, d_values_, nodes_, alpha, beta);
        if (!cuda_ok(cudaGetLastError()) ||
            !cuda_ok(cub::DeviceReduce::Sum(
                d_reduce_temp_, reduce_temp_bytes_, d_values_, d_reduce_result_, nodes_)) ||
            !cuda_ok(cudaMemcpy(&residual, d_reduce_result_, sizeof(double), cudaMemcpyDeviceToHost))) {
            return make_result(KatzStatus::cuda_error, iterations, residual);
        }
        ++iterations;
        std::swap(d_current_, d_next_);
        if (!std::isfinite(residual)) break;
        if (residual < epsilon) {
            converged = true;
            break;
        }
    }
    if (!converged) return make_result(KatzStatus::not_converged, iterations, residual);

    if (normalized) {
        double norm_sq = 0.0;
        square_scores_kernel<<<blocks, kBlockSize>>>(d_current_, d_values_, nodes_);
        if (!cuda_ok(cudaGetLastError()) ||
            !cuda_ok(cub::DeviceReduce::Sum(
                d_reduce_temp_, reduce_temp_bytes_, d_values_, d_reduce_result_, nodes_)) ||
            !cuda_ok(cudaMemcpy(&norm_sq, d_reduce_result_, sizeof(double), cudaMemcpyDeviceToHost))) {
            return make_result(KatzStatus::cuda_error, iterations, residual);
        }
        const double norm = std::sqrt(norm_sq);
        if (!(norm > 0.0) || !std::isfinite(norm)) {
            return make_result(KatzStatus::invalid_input, iterations, residual);
        }
        normalize_kernel<<<blocks, kBlockSize>>>(d_current_, nodes_, norm);
        if (!cuda_ok(cudaGetLastError())) return make_result(KatzStatus::cuda_error, iterations, residual);
    }

    scores.assign(nodes_, 0.0);
    const bool copied = cuda_ok(cudaMemcpy(scores.data(), d_current_,
                                            scores.size() * sizeof(double), cudaMemcpyDeviceToHost));
    return copied ? make_result(KatzStatus::success, iterations, residual)
                  : make_result(KatzStatus::cuda_error, iterations, residual);
}

KatzResult katz_centrality(const std::vector<int>& incoming_offsets,
                           const std::vector<int>& incoming_columns,
                           double alpha,
                           double beta,
                           int max_iter,
                           double epsilon,
                           bool normalized,
                           std::vector<double>& scores) {
    KatzStatus status = KatzStatus::invalid_input;
    std::unique_ptr<PreparedKatzContext> context =
        PreparedKatzContext::create(incoming_offsets, incoming_columns, status);
    if (!context) return make_result(status, 0, 0.0);
    return context->run(alpha, beta, max_iter, epsilon, normalized, scores);
}

}  // namespace gpu_easygraph
