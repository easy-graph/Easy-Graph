#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "../../cpp_easygraph/distributed/communicator.h"

int main() {
    egdist::Communicator comm;
    const int rank = comm.rank();
    const int size = comm.size();
    std::vector<std::vector<std::uint64_t>> buckets(size);
    for (int destination = 0; destination < size; ++destination) {
        const int count = rank == 0 && destination == 0 ? 0 : (rank + 1) * 13 + destination;
        for (int index = 0; index < count; ++index) {
            buckets[destination].push_back(rank * 100000 + destination * 1000 + index);
        }
    }

    // Four uint64_t records per destination per round with two ranks.
    const auto received = comm.exchange(buckets, 64);
    std::vector<std::uint64_t> expected;
    for (int source = 0; source < size; ++source) {
        const int count = source == 0 && rank == 0 ? 0 : (source + 1) * 13 + rank;
        for (int index = 0; index < count; ++index) {
            expected.push_back(source * 100000 + rank * 1000 + index);
        }
    }
    comm.check_error(received == expected ? "" : "Chunked exchange changed record order or values");
    if (rank == 0) std::cout << "chunked MPI exchange: OK" << std::endl;
    comm.close();
    egdist::finalize_mpi();
}
