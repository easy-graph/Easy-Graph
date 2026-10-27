#include "communicator.h"

#include <algorithm>
#include <cstdlib>

#include "../common/pagerank_parallel.h"

namespace egdist {
namespace {
bool owns_mpi = false;
}

void mpi_check(int code) {
    if (code == MPI_SUCCESS) return;
    char message[MPI_MAX_ERROR_STRING];
    int length = 0;
    MPI_Error_string(code, message, &length);
    throw std::runtime_error(std::string(message, length));
}

Communicator::Communicator() {
    if (const char* requested = std::getenv("OMP_NUM_THREADS")) {
        std::size_t consumed = 0;
        const std::string value(requested);
        const int threads = std::stoi(value, &consumed);
        if (consumed != value.size()) {
            throw std::invalid_argument("OMP_NUM_THREADS must be a positive integer");
        }
        easygraph_parallel::configure(threads);
    }
    int initialized = 0, finalized = 0;
    MPI_Initialized(&initialized);
    MPI_Finalized(&finalized);
    if (finalized) throw std::runtime_error("MPI has already been finalized");
    if (!initialized) {
        int provided = 0;
        mpi_check(MPI_Init_thread(nullptr, nullptr, MPI_THREAD_FUNNELED, &provided));
        owns_mpi = true;
        if (provided < MPI_THREAD_FUNNELED) {
            throw std::runtime_error("MPI_THREAD_FUNNELED is required");
        }
    }
    mpi_check(MPI_Comm_dup(MPI_COMM_WORLD, &comm_));
    mpi_check(MPI_Comm_set_errhandler(comm_, MPI_ERRORS_RETURN));
    mpi_check(MPI_Comm_rank(comm_, &rank_));
    mpi_check(MPI_Comm_size(comm_, &size_));
}

Communicator::~Communicator() {
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized && !closed()) MPI_Comm_free(&comm_);
}

void finalize_mpi() {
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (owns_mpi && !finalized) MPI_Finalize();
}

void Communicator::ensure_open() const {
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (closed() || finalized) throw std::runtime_error("The distributed Context is closed");
}

void Communicator::close() {
    if (closed()) return;
    ensure_open();
    mpi_check(MPI_Comm_free(&comm_));
}

void Communicator::barrier() const { mpi_check(MPI_Barrier(handle())); }

void Communicator::check_error(const std::string& error) const {
    int candidate = error.empty() ? size_ : rank_, source = size_;
    mpi_check(MPI_Allreduce(&candidate, &source, 1, MPI_INT, MPI_MIN, handle()));
    if (source == size_) return;
    int length = rank_ == source ? static_cast<int>(error.size()) : 0;
    mpi_check(MPI_Bcast(&length, 1, MPI_INT, source, handle()));
    std::string message = rank_ == source ? error : std::string(length, '\0');
    mpi_check(MPI_Bcast(&message[0], length, MPI_CHAR, source, handle()));
    throw std::runtime_error("rank " + std::to_string(source) + ": " + message);
}

void Communicator::agree(const std::string& value) const {
    int length = rank_ == 0 ? static_cast<int>(value.size()) : 0;
    mpi_check(MPI_Bcast(&length, 1, MPI_INT, 0, handle()));
    std::string expected = rank_ == 0 ? value : std::string(length, '\0');
    mpi_check(MPI_Bcast(length ? &expected[0] : nullptr, length, MPI_CHAR, 0, handle()));
    check_error(value == expected ? "" : "Collective arguments differ between ranks");
}

std::string Communicator::broadcast_string(const std::string& value, int root) const {
    agree(std::to_string(root));
    check_error(root < 0 || root >= size_ ? "Invalid broadcast root" : "");
    int length = rank_ == root ? static_cast<int>(value.size()) : 0;
    mpi_check(MPI_Bcast(&length, 1, MPI_INT, root, handle()));
    std::string result = rank_ == root ? value : std::string(length, '\0');
    mpi_check(MPI_Bcast(length ? &result[0] : nullptr, length, MPI_CHAR, root, handle()));
    return result;
}

double Communicator::sum(double value) const {
    double result = 0;
    mpi_check(MPI_Allreduce(&value, &result, 1, MPI_DOUBLE, MPI_SUM, handle()));
    return result;
}

Offset Communicator::sum(Offset value) const {
    Offset result = 0;
    mpi_check(MPI_Allreduce(&value, &result, 1, MPI_UINT64_T, MPI_SUM, handle()));
    return result;
}

std::pair<std::vector<VertexId>, std::vector<double>> Communicator::gather_result(
    const std::vector<VertexId>& ids, const std::vector<double>& values, int root) const {
    agree(std::to_string(root));
    check_error(root < 0 || root >= size_ ? "Invalid gather root" :
                ids.size() != values.size() ? "Result array lengths differ" :
                ids.size() > INT_MAX ? "Local result exceeds MPI count limit" : "");
    int count = static_cast<int>(ids.size());
    std::vector<int> counts(size_), offsets(size_);
    mpi_check(MPI_Gather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT, root, handle()));
    std::size_t total = 0;
    bool overflow = false;
    if (rank_ == root) {
        for (int i = 0; i < size_; ++i) {
            overflow = overflow || total > static_cast<std::size_t>(INT_MAX - counts[i]);
            if (!overflow) { offsets[i] = static_cast<int>(total); total += counts[i]; }
        }
    }
    check_error(overflow ? "Gathered result exceeds MPI count limit" : "");
    std::pair<std::vector<VertexId>, std::vector<double>> result;
    result.first.resize(total);
    result.second.resize(total);
    mpi_check(MPI_Gatherv(ids.data(), count, MPI_INT64_T, result.first.data(), counts.data(),
                         offsets.data(), MPI_INT64_T, root, handle()));
    mpi_check(MPI_Gatherv(values.data(), count, MPI_DOUBLE, result.second.data(), counts.data(),
                         offsets.data(), MPI_DOUBLE, root, handle()));
    return result;
}
}
