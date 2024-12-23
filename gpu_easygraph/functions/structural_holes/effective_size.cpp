#include <string>
#include <vector>
#include <memory> 

#include "structural_holes/effective_size.cuh"
#include "common.h"

namespace gpu_easygraph {

using std::vector;

int effective_size(
    _IN_ const vector<int>& V,
    _IN_ const vector<int>& E,
    _IN_ const vector<int>& row,
    _IN_ const vector<int>& col,
    _IN_ int num_nodes,
    _IN_ const vector<double>& W,
    _IN_ bool is_directed,
    _IN_ vector<int>& node_mask,  
    _OUT_ vector<double>& effective_size
) {
    int num_edges = row.size();
    
    effective_size = vector<double>(num_nodes);
    int r = cuda_effective_size(V.data(), E.data(), row.data(), col.data(), W.data(), num_nodes, num_edges, is_directed, node_mask.data(), effective_size.data());

    return r; 
}

} // namespace gpu_easygraph