#include <string>
#include <vector>
#include <memory> // 需要包含memory头文件

#include "structural_holes/evaluation.cuh"
#include "common.h"

namespace gpu_easygraph {

using std::vector;

// static int decide_warp_size(
//     _IN_ int len_V,
//     _IN_ int len_E
// )
// {
//     vector<int> warp_size_cand{1, 2, 4, 8, 16, 32};

//     if (len_E / len_V < warp_size_cand.front()) {
//         return warp_size_cand.front();
//     }

//     for (int i = 0; i + 1 < warp_size_cand.size(); ++i) {
//         if (warp_size_cand[i] <= len_E / len_V
//                 && len_E / len_V < warp_size_cand[i + 1]) {
//             return warp_size_cand[i + 1];
//         }
//     }
//     return warp_size_cand.back();
// }

int effective_size(
    _IN_ const vector<int>& V,
    _IN_ const vector<int>& E,
    _IN_ const vector<int>& row,
    _IN_ const vector<int>& col,
    _IN_ int num_nodes,
    _IN_ const vector<double>& W,
    _IN_ bool is_directed,
    _IN_ vector<int>& node_mask,  // 添加节点掩码参数
    _OUT_ vector<double>& effective_size
) {
    int num_edges = row.size();
    
    effective_size = vector<double>(num_nodes);
    int r = cuda_effective_size(V.data(), E.data(), row.data(), col.data(), W.data(), num_nodes, num_edges, is_directed, node_mask.data(), effective_size.data());

    return r;  // 成功
}

int constraint(
    _IN_ const vector<int>& V,
    _IN_ const vector<int>& E,
    _IN_ const vector<int>& row,
    _IN_ const vector<int>& col,
    _IN_ int num_nodes,
    _IN_ const vector<double>& W,
    _IN_ bool is_directed,
    _IN_ vector<int>& node_mask,  // 添加节点掩码参数
    _OUT_ vector<double>& constraint
) {
    int num_edges = row.size();
    
    constraint = vector<double>(num_nodes);
    int r = cuda_constraint(V.data(), E.data(), row.data(), col.data(), W.data(), num_nodes, num_edges, is_directed, node_mask.data(), constraint.data());

    return r;  // 成功
}

int hierarchy(
    _IN_ const vector<int>& V,
    _IN_ const vector<int>& E,
    _IN_ const vector<int>& row,
    _IN_ const vector<int>& col,
    _IN_ int num_nodes,
    _IN_ const vector<double>& W,
    _IN_ bool is_directed,
    _IN_ vector<int>& node_mask,  // 添加节点掩码参数
    _OUT_ vector<double>& hierarchy
) {
    int num_edges = row.size();
    
    hierarchy = vector<double>(num_nodes);
    int r = cuda_hierarchy(V.data(), E.data(), row.data(), col.data(), W.data(), num_nodes, num_edges, is_directed, node_mask.data(), hierarchy.data());

    return r;  // 成功
}

} // namespace gpu_easygraph