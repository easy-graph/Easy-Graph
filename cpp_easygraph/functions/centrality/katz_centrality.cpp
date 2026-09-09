#ifdef _OPENMP
#include <omp.h>
#endif
#include <cmath>
#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#if defined(EASYGRAPH_ENABLE_GPU) && !defined(_WIN32)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "centrality.h"
#include "../../classes/graph.h"
#ifdef EASYGRAPH_ENABLE_GPU
#include <gpu_easygraph.h>
#endif

namespace py = pybind11;

#ifdef EASYGRAPH_ENABLE_GPU
namespace {

constexpr const char* kPreparedGpuKatzCapsule = "easygraph.PreparedGpuKatzContext";

struct PreparedGpuKatzHandle {
    std::unique_ptr<gpu_easygraph::PreparedKatzContext> context;
    std::vector<py::object> node_labels;
};

void destroy_prepared_gpu_katz_context(PyObject* capsule) {
    void* pointer = PyCapsule_GetPointer(capsule, kPreparedGpuKatzCapsule);
    if (pointer) delete static_cast<PreparedGpuKatzHandle*>(pointer);
}

PreparedGpuKatzHandle* prepared_gpu_katz_handle(const py::object& capsule) {
    if (!PyCapsule_IsValid(capsule.ptr(), kPreparedGpuKatzCapsule)) {
        throw py::value_error("invalid or closed PreparedKatzContext");
    }
    return static_cast<PreparedGpuKatzHandle*>(
        PyCapsule_GetPointer(capsule.ptr(), kPreparedGpuKatzCapsule));
}

void throw_katz_error(const gpu_easygraph::KatzResult& run, int max_iter) {
    if (run.status == gpu_easygraph::KatzStatus::not_converged) {
        throw std::runtime_error(
            "EGGPU deterministic Katz did not converge within max_iter=" +
            std::to_string(max_iter) + "; final L1 residual=" + std::to_string(run.final_residual));
    }
    if (run.status == gpu_easygraph::KatzStatus::invalid_input) {
        throw py::value_error(
            "EGGPU deterministic Katz requires finite alpha/beta, epsilon > 0, and "
            "alpha < 1 / max_in_degree for a guaranteed-convergent incoming CSR iteration");
    }
    if (run.status != gpu_easygraph::KatzStatus::success) {
        throw std::runtime_error("EGGPU deterministic Katz CUDA execution failed");
    }
}

}  // namespace
#endif

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

#ifdef EASYGRAPH_ENABLE_GPU
namespace {

constexpr const char* kGpuKatzTsvDatasetCapsule = "easygraph.GpuKatzTsvDataset";

struct GpuKatzTsvDatasetHandle {
    std::vector<int> sources;
    std::vector<int> targets;
    std::vector<int> incoming_offsets;
    std::vector<int> incoming_columns;
    std::vector<long long> original_ids;
    int max_in_degree = 0;
};

void destroy_gpu_katz_tsv_dataset(PyObject* capsule) {
    void* pointer = PyCapsule_GetPointer(capsule, kGpuKatzTsvDatasetCapsule);
    if (pointer) delete static_cast<GpuKatzTsvDatasetHandle*>(pointer);
}

GpuKatzTsvDatasetHandle* gpu_katz_tsv_dataset_handle(const py::object& capsule) {
    if (!PyCapsule_IsValid(capsule.ptr(), kGpuKatzTsvDatasetCapsule)) {
        throw py::value_error("invalid or closed GPU Katz TSV dataset");
    }
    return static_cast<GpuKatzTsvDatasetHandle*>(
        PyCapsule_GetPointer(capsule.ptr(), kGpuKatzTsvDatasetCapsule));
}

#if !defined(_WIN32)
struct MappedFile {
    int fd = -1;
    const char* data = nullptr;
    size_t size = 0;

    explicit MappedFile(const std::string& path) {
        fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("cannot open arcs TSV: " + path);
        struct stat info {};
        if (fstat(fd, &info) != 0 || info.st_size < 0) {
            close(fd);
            fd = -1;
            throw std::runtime_error("cannot stat arcs TSV: " + path);
        }
        size = static_cast<size_t>(info.st_size);
        if (size == 0) return;
        data = static_cast<const char*>(mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0));
        if (data == MAP_FAILED) {
            close(fd);
            fd = -1;
            data = nullptr;
            throw std::runtime_error("cannot mmap arcs TSV: " + path);
        }
    }

    ~MappedFile() {
        if (data) munmap(const_cast<char*>(data), size);
        if (fd >= 0) close(fd);
    }

    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;
};

bool parse_nonnegative_int(const char*& cursor, const char* end, int& value) {
    while (cursor < end && (*cursor == ' ' || *cursor == '\t' || *cursor == '\r' ||
                            *cursor == '\n')) {
        ++cursor;
    }
    if (cursor == end) return false;
    if (*cursor < '0' || *cursor > '9') {
        throw std::runtime_error("arcs TSV contains a non-integer token");
    }
    std::int64_t parsed = 0;
    while (cursor < end && *cursor >= '0' && *cursor <= '9') {
        parsed = parsed * 10 + (*cursor - '0');
        if (parsed > std::numeric_limits<int>::max()) {
            throw std::runtime_error("arcs TSV node ID exceeds int32 range");
        }
        ++cursor;
    }
    value = static_cast<int>(parsed);
    return true;
}

long long parse_integer_field(const std::string& field, const char* error_message) {
    if (field.empty()) throw std::runtime_error(error_message);
    errno = 0;
    char* parsed_end = nullptr;
    const long long value = std::strtoll(field.c_str(), &parsed_end, 10);
    if (errno == ERANGE || parsed_end == field.c_str() || *parsed_end != '\0') {
        throw std::runtime_error(error_message);
    }
    return value;
}

std::vector<long long> load_node_map(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open node map TSV: " + path);
    std::string line;
    if (!std::getline(input, line)) {
        throw std::runtime_error("node map must begin with new_id<TAB>original_id");
    }
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line != "new_id\toriginal_id") {
        throw std::runtime_error("node map must begin with new_id<TAB>original_id");
    }

    std::vector<long long> original_ids;
    std::unordered_set<long long> seen_original_ids;
    int expected_id = 0;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const size_t tab = line.find('\t');
        if (tab == std::string::npos) throw std::runtime_error("malformed node map row");
        const long long new_id = parse_integer_field(
            line.substr(0, tab), "node map new_id values must be contiguous from zero");
        if (new_id != expected_id) {
            throw std::runtime_error("node map new_id values must be contiguous from zero");
        }
        const long long original_id = parse_integer_field(
            line.substr(tab + 1), "node map original_id values must be integers");
        if (!seen_original_ids.insert(original_id).second) {
            throw std::runtime_error("node map original_id values must be unique");
        }
        original_ids.push_back(original_id);
        ++expected_id;
    }
    return original_ids;
}

std::unique_ptr<GpuKatzTsvDatasetHandle> load_gpu_katz_tsv_dataset(
    const std::string& arcs_path, const std::string& node_map_path) {
    std::unique_ptr<GpuKatzTsvDatasetHandle> dataset(new GpuKatzTsvDatasetHandle());
    dataset->original_ids = load_node_map(node_map_path);

    if (dataset->original_ids.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("node map has too many nodes for EGGPU int32 CSR");
    }
    MappedFile file(arcs_path);
    if (file.size > 0) {
        dataset->sources.reserve(file.size / 12);
        dataset->targets.reserve(file.size / 12);
        const char* cursor = file.data;
        const char* const end = file.data + file.size;
        while (cursor < end) {
            int source = 0;
            if (!parse_nonnegative_int(cursor, end, source)) break;
            int target = 0;
            if (!parse_nonnegative_int(cursor, end, target)) {
                throw std::runtime_error("arcs TSV ends after a source ID");
            }
            while (cursor < end && (*cursor == ' ' || *cursor == '\t' || *cursor == '\r')) {
                ++cursor;
            }
            if (cursor < end && *cursor != '\n') {
                throw std::runtime_error("arcs TSV rows must contain exactly two integer columns");
            }
            if (cursor < end) ++cursor;
            dataset->sources.push_back(source);
            dataset->targets.push_back(target);
        }
    }
    for (size_t edge = 0; edge < dataset->sources.size(); ++edge) {
        if (dataset->sources[edge] >= static_cast<int>(dataset->original_ids.size()) ||
            dataset->targets[edge] >= static_cast<int>(dataset->original_ids.size())) {
            throw std::runtime_error("arcs TSV endpoint is outside node map range");
        }
    }
    if (dataset->sources.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("arcs TSV has too many edges for EGGPU int32 CSR");
    }

    const int nodes = static_cast<int>(dataset->original_ids.size());
    dataset->incoming_offsets.assign(static_cast<size_t>(nodes) + 1, 0);
    for (int target : dataset->targets) ++dataset->incoming_offsets[static_cast<size_t>(target) + 1];
    for (int node = 1; node <= nodes; ++node) {
        dataset->incoming_offsets[node] += dataset->incoming_offsets[node - 1];
    }
    dataset->incoming_columns.resize(dataset->sources.size());
    std::vector<int> write_positions = dataset->incoming_offsets;
    for (size_t edge = 0; edge < dataset->sources.size(); ++edge) {
        dataset->incoming_columns[write_positions[dataset->targets[edge]]++] = dataset->sources[edge];
    }
    for (int node = 0; node < nodes; ++node) {
        dataset->max_in_degree = std::max(
            dataset->max_in_degree, dataset->incoming_offsets[node + 1] - dataset->incoming_offsets[node]);
    }
    return dataset;
}
#endif

}  // namespace
#endif

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

py::object cpp_gpu_katz_centrality(py::object G,
                                   py::object py_alpha,
                                   py::object py_beta,
                                   py::object py_max_iter,
                                   py::object py_tol,
                                   py::object py_normalized) {
#ifndef EASYGRAPH_ENABLE_GPU
    (void)G;
    (void)py_alpha;
    (void)py_beta;
    (void)py_max_iter;
    (void)py_tol;
    (void)py_normalized;
    throw std::runtime_error("GPU Katz requires EasyGraph to be built with EASYGRAPH_ENABLE_GPU=TRUE");
#else
    Graph& graph = G.cast<Graph&>();
    if (!py::isinstance<py::float_>(py_beta) && !py::isinstance<py::int_>(py_beta)) {
        throw py::type_error("GPU Katz currently requires scalar beta");
    }
    const double alpha = py_alpha.cast<double>();
    const double beta = py_beta.cast<double>();
    const int max_iter = py_max_iter.cast<int>();
    const double tol = py_tol.cast<double>();
    const bool normalized = py_normalized.cast<bool>();
    if (max_iter < 1 || tol <= 0.0) {
        throw py::value_error("max_iter and tol must be positive");
    }

    std::shared_ptr<CSRGraph> csr_ptr = graph.gen_CSR();
    if (!csr_ptr || csr_ptr->nodes.empty()) {
        return py::dict();
    }
    CSRMatrix incoming = build_transpose_matrix_from_csr(csr_ptr);
    std::vector<double> scores;
    const auto run = gpu_easygraph::katz_centrality(
        incoming.indptr, incoming.indices, alpha, beta, max_iter, tol, normalized, scores);
    throw_katz_error(run, max_iter);

    py::dict result;
    for (int i = 0; i < incoming.rows; ++i) {
        node_t internal_id = csr_ptr->nodes[i];
        result[graph.id_to_node[py::cast(internal_id)]] = scores[i];
    }
    return result;
#endif
}

py::object cpp_prepare_gpu_katz(py::object G) {
#ifndef EASYGRAPH_ENABLE_GPU
    (void)G;
    throw std::runtime_error("Prepared GPU Katz requires EasyGraph to be built with EASYGRAPH_ENABLE_GPU=TRUE");
#else
    Graph& graph = G.cast<Graph&>();
    std::unique_ptr<PreparedGpuKatzHandle> handle(new PreparedGpuKatzHandle());
    std::shared_ptr<CSRGraph> csr_ptr = graph.gen_CSR();
    if (csr_ptr && !csr_ptr->nodes.empty()) {
        CSRMatrix incoming = build_transpose_matrix_from_csr(csr_ptr);
        gpu_easygraph::KatzStatus status = gpu_easygraph::KatzStatus::invalid_input;
        handle->context = gpu_easygraph::PreparedKatzContext::create(
            incoming.indptr, incoming.indices, status);
        if (!handle->context) {
            if (status == gpu_easygraph::KatzStatus::invalid_input) {
                throw py::value_error("cannot prepare EGGPU Katz from the graph incoming CSR");
            }
            throw std::runtime_error("EGGPU prepared Katz CUDA initialization failed");
        }
        handle->node_labels.reserve(static_cast<size_t>(incoming.rows));
        for (int i = 0; i < incoming.rows; ++i) {
            node_t internal_id = csr_ptr->nodes[i];
            handle->node_labels.push_back(graph.id_to_node[py::cast(internal_id)]);
        }
    }
    return py::capsule(handle.release(), kPreparedGpuKatzCapsule, destroy_prepared_gpu_katz_context);
#endif
}

py::object cpp_run_prepared_gpu_katz(py::object py_context,
                                     py::object py_alpha,
                                     py::object py_beta,
                                     py::object py_max_iter,
                                     py::object py_tol,
                                     py::object py_normalized) {
#ifndef EASYGRAPH_ENABLE_GPU
    (void)py_context;
    (void)py_alpha;
    (void)py_beta;
    (void)py_max_iter;
    (void)py_tol;
    (void)py_normalized;
    throw std::runtime_error("Prepared GPU Katz requires EasyGraph to be built with EASYGRAPH_ENABLE_GPU=TRUE");
#else
    PreparedGpuKatzHandle* handle = prepared_gpu_katz_handle(py_context);
    if (!py::isinstance<py::float_>(py_beta) && !py::isinstance<py::int_>(py_beta)) {
        throw py::type_error("GPU Katz currently requires scalar beta");
    }
    const double alpha = py_alpha.cast<double>();
    const double beta = py_beta.cast<double>();
    const int max_iter = py_max_iter.cast<int>();
    const double tol = py_tol.cast<double>();
    const bool normalized = py_normalized.cast<bool>();
    if (max_iter < 1 || tol <= 0.0) {
        throw py::value_error("max_iter and tol must be positive");
    }
    if (!handle->context) return py::dict();

    std::vector<double> scores;
    const auto run = handle->context->run(alpha, beta, max_iter, tol, normalized, scores);
    throw_katz_error(run, max_iter);
    if (scores.size() != handle->node_labels.size()) {
        throw std::runtime_error("EGGPU prepared Katz returned an invalid score vector");
    }

    py::dict result;
    for (size_t i = 0; i < scores.size(); ++i) {
        result[handle->node_labels[i]] = scores[i];
    }
    return result;
#endif
}

py::object cpp_load_gpu_katz_tsv_dataset(py::object py_arcs_path, py::object py_node_map_path) {
#ifndef EASYGRAPH_ENABLE_GPU
    (void)py_arcs_path;
    (void)py_node_map_path;
    throw std::runtime_error("GPU Katz TSV loading requires EASYGRAPH_ENABLE_GPU=TRUE");
#elif defined(_WIN32)
    (void)py_arcs_path;
    (void)py_node_map_path;
    throw std::runtime_error("GPU Katz TSV mmap loading is currently available on POSIX platforms only");
#else
    const std::string arcs_path = py_arcs_path.cast<std::string>();
    const std::string node_map_path = py_node_map_path.cast<std::string>();
    std::unique_ptr<GpuKatzTsvDatasetHandle> dataset =
        load_gpu_katz_tsv_dataset(arcs_path, node_map_path);
    return py::capsule(dataset.release(), kGpuKatzTsvDatasetCapsule, destroy_gpu_katz_tsv_dataset);
#endif
}

py::object cpp_gpu_katz_tsv_dataset_to_digraph(py::object py_dataset) {
#ifndef EASYGRAPH_ENABLE_GPU
    (void)py_dataset;
    throw std::runtime_error("GPU Katz TSV loading requires EASYGRAPH_ENABLE_GPU=TRUE");
#else
    GpuKatzTsvDatasetHandle* dataset = gpu_katz_tsv_dataset_handle(py_dataset);
    py::object graph = py::module_::import("easygraph").attr("DiGraphC")();
    constexpr size_t kBatchSize = 65536;

    for (size_t begin = 0; begin < dataset->original_ids.size(); begin += kBatchSize) {
        const size_t end = std::min(begin + kBatchSize, dataset->original_ids.size());
        py::list nodes;
        for (size_t index = begin; index < end; ++index) {
            nodes.append(py::int_(dataset->original_ids[index]));
        }
        graph.attr("add_nodes")(nodes);
    }
    for (size_t begin = 0; begin < dataset->sources.size(); begin += kBatchSize) {
        const size_t end = std::min(begin + kBatchSize, dataset->sources.size());
        py::list edges;
        for (size_t index = begin; index < end; ++index) {
            edges.append(py::make_tuple(
                dataset->original_ids[dataset->sources[index]],
                dataset->original_ids[dataset->targets[index]]));
        }
        graph.attr("add_edges")(edges);
    }
    return graph;
#endif
}

py::object cpp_prepare_gpu_katz_tsv_dataset(py::object py_dataset) {
#ifndef EASYGRAPH_ENABLE_GPU
    (void)py_dataset;
    throw std::runtime_error("Prepared GPU Katz requires EasyGraph to be built with EASYGRAPH_ENABLE_GPU=TRUE");
#else
    GpuKatzTsvDatasetHandle* dataset = gpu_katz_tsv_dataset_handle(py_dataset);
    std::unique_ptr<PreparedGpuKatzHandle> handle(new PreparedGpuKatzHandle());
    if (!dataset->original_ids.empty()) {
        gpu_easygraph::KatzStatus status = gpu_easygraph::KatzStatus::invalid_input;
        handle->context = gpu_easygraph::PreparedKatzContext::create(
            dataset->incoming_offsets, dataset->incoming_columns, status);
        if (!handle->context) {
            if (status == gpu_easygraph::KatzStatus::invalid_input) {
                throw py::value_error("cannot prepare EGGPU Katz from the TSV incoming CSR");
            }
            throw std::runtime_error("EGGPU prepared Katz CUDA initialization failed");
        }
        handle->node_labels.reserve(dataset->original_ids.size());
        for (long long original_id : dataset->original_ids) {
            handle->node_labels.push_back(py::int_(original_id));
        }
    }
    return py::capsule(handle.release(), kPreparedGpuKatzCapsule, destroy_prepared_gpu_katz_context);
#endif
}

py::object cpp_gpu_katz_tsv_dataset_metadata(py::object py_dataset) {
#ifndef EASYGRAPH_ENABLE_GPU
    (void)py_dataset;
    throw std::runtime_error("GPU Katz TSV loading requires EASYGRAPH_ENABLE_GPU=TRUE");
#else
    GpuKatzTsvDatasetHandle* dataset = gpu_katz_tsv_dataset_handle(py_dataset);
    return py::make_tuple(
        dataset->original_ids.size(), dataset->sources.size(), dataset->max_in_degree);
#endif
}
