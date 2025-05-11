#include <string>
#include <vector>
#include <memory>

#include "structural_holes/constraint.cuh"
#include "common.h"

namespace gpu_easygraph {

using std::vector;

int constraint(
    _IN_ const vector<int>& V,
    _IN_ const vector<int>& E,
    _IN_ const vector<int>& row,
    _IN_ const vector<int>& col,
    _IN_ int num_nodes,
    _IN_ const vector<double>& W,
    _IN_ bool is_directed,
    _IN_ vector<int>& node_mask,
    _OUT_ vector<double>& constraint
) {
    int num_edges = row.size();
    
    constraint = vector<double>(num_nodes);
    int r = cuda_constraint(V.data(), E.data(), row.data(), col.data(), W.data(), num_nodes, num_edges, is_directed, node_mask.data(), constraint.data());

    return r;
}

} // namespace gpu_easygraph