#pragma once

#include <mpi.h>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "types.h"

namespace egdist {
void mpi_check(int code);
void finalize_mpi();

class Communicator {
public:
    Communicator();
    ~Communicator();
    Communicator(const Communicator&) = delete;
    Communicator& operator=(const Communicator&) = delete;
    void close();
    bool closed() const { return comm_ == MPI_COMM_NULL; }
    void ensure_open() const;
    int rank() const { ensure_open(); return rank_; }
    int size() const { ensure_open(); return size_; }
    MPI_Comm handle() const { ensure_open(); return comm_; }
    void barrier() const;
    void check_error(const std::string& error) const;
    void agree(const std::string& value) const;
    std::string broadcast_string(const std::string& value, int root) const;
    double sum(double value) const;
    Offset sum(Offset value) const;
    std::pair<std::vector<VertexId>, std::vector<double>> gather_result(
        const std::vector<VertexId>& ids, const std::vector<double>& values,
        int root) const;



    template <typename T>
    std::vector<T> exchange(const std::vector<std::vector<T>>& buckets,
                            std::size_t max_round_bytes = 256u << 20) const {
        static_assert(std::is_trivially_copyable<T>::value, "POD records required");
        ensure_open();
        check_error(buckets.size() == static_cast<std::size_t>(size_) ? "" :
                    "Incorrect number of destination buckets");
        // MPI_Alltoallv uses int byte counts and displacements. Exchange the full
        // element counts first, then move bounded rounds into the final buffer.
        std::vector<unsigned long long> send_sizes(size_), recv_sizes(size_);
        for (int i = 0; i < size_; ++i) send_sizes[i] = buckets[i].size();
        mpi_check(MPI_Alltoall(send_sizes.data(), 1, MPI_UNSIGNED_LONG_LONG,
                               recv_sizes.data(), 1, MPI_UNSIGNED_LONG_LONG, handle()));

        std::vector<std::size_t> recv_starts(size_), recv_written(size_, 0);
        std::size_t recv_total = 0;
        bool overflow = false;
        for (int i = 0; i < size_; ++i) {
            overflow = overflow || recv_sizes[i] >
                (std::numeric_limits<std::size_t>::max() - recv_total);
            if (!overflow) {
                recv_starts[i] = recv_total;
                recv_total += static_cast<std::size_t>(recv_sizes[i]);
            }
        }
        check_error(overflow ? "MPI exchange receive size overflow" : "");
        std::vector<T> received(recv_total);

        const auto round_bytes = std::min(max_round_bytes, static_cast<std::size_t>(INT_MAX));
        const auto round_records = round_bytes / static_cast<std::size_t>(size_) / sizeof(T);
        check_error(round_records ? "" : "MPI communicator has too many ranks for exchange");
        std::vector<std::size_t> send_written(size_, 0);
        std::vector<int> counts(size_), offsets(size_), recv_counts(size_), recv_offsets(size_);
        while (true) {
            int pending = 0;
            for (int i = 0; i < size_; ++i) {
                pending |= send_written[i] < buckets[i].size();
            }
            int any_pending = 0;
            mpi_check(MPI_Allreduce(&pending, &any_pending, 1, MPI_INT, MPI_MAX, handle()));
            if (!any_pending) break;

            std::size_t send_round_total = 0;
            for (int i = 0; i < size_; ++i) {
                offsets[i] = static_cast<int>(send_round_total * sizeof(T));
                const auto count = std::min(round_records, buckets[i].size() - send_written[i]);
                counts[i] = static_cast<int>(count * sizeof(T));
                send_round_total += count;
            }
            std::vector<T> send;
            send.reserve(send_round_total);
            for (int i = 0; i < size_; ++i) {
                const auto count = static_cast<std::size_t>(counts[i]) / sizeof(T);
                send.insert(send.end(), buckets[i].begin() + send_written[i],
                            buckets[i].begin() + send_written[i] + count);
                send_written[i] += count;
            }
            mpi_check(MPI_Alltoall(counts.data(), 1, MPI_INT, recv_counts.data(), 1,
                                   MPI_INT, handle()));
            std::size_t recv_round_total = 0;
            for (int i = 0; i < size_; ++i) {
                recv_offsets[i] = static_cast<int>(recv_round_total * sizeof(T));
                recv_round_total += static_cast<std::size_t>(recv_counts[i]) / sizeof(T);
            }
            std::vector<T> recv_round(recv_round_total);
            mpi_check(MPI_Alltoallv(send.data(), counts.data(), offsets.data(), MPI_BYTE,
                                    recv_round.data(), recv_counts.data(), recv_offsets.data(),
                                    MPI_BYTE, handle()));
            for (int i = 0; i < size_; ++i) {
                const auto count = static_cast<std::size_t>(recv_counts[i]) / sizeof(T);
                if (count) {
                    std::copy_n(recv_round.begin() + recv_offsets[i] / sizeof(T), count,
                                received.begin() + recv_starts[i] + recv_written[i]);
                }
                recv_written[i] += count;
            }
        }
        return received;
    }

private:
    MPI_Comm comm_ = MPI_COMM_NULL;
    int rank_ = 0;
    int size_ = 0;
};
}
