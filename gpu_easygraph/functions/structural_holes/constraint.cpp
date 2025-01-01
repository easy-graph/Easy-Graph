#include <string>
#include <vector>
#include <memory>

#include "structural_holes/constraint.cuh"
#include "common.h"

namespace gpu_easygraph {

using std::vector;

int constraint(
    _IN_ int num_nodes,
    _IN_ const std::vector<int>& rowPtrOut,
    _IN_ const std::vector<int>& colIdxOut,
    _IN_ const std::vector<double>& valOut,
    _IN_ const std::vector<int>& rowPtrIn,
    _IN_ const std::vector<int>& colIdxIn,
    _IN_ const std::vector<double>& valIn,
    _IN_ bool is_directed,
    _IN_ vector<int>& node_mask,
    _OUT_ vector<double>& constraints
) {
    int len_rowPtrOut = rowPtrOut.size();
    int len_colIdxOut = colIdxOut.size();
    
    constraints = vector<double>(num_nodes);
    int r = cuda_constraint(num_nodes, len_rowPtrOut, len_colIdxOut, rowPtrOut.data(), colIdxOut.data(), valOut.data(), rowPtrIn.data(), colIdxIn.data(), valIn.data(), is_directed, node_mask.data(), constraints.data());

    return r;
}

} // namespace gpu_easygraph