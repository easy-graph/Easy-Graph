#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>
#ifdef _OPENMP
#include <omp.h>
#else
#warning "OpenMP is not available: modularity utility functions will fall back to single-threaded execution."
#endif

#include "../../classes/graph.h"
#include "../../common/utils.h"

using namespace std;

void addVectorsInPlace(std::vector<double>& v1, std::vector<double>& v2) {
    if (v1.size() != v2.size()) {
        throw std::invalid_argument("Vectors must have the same size for element-wise addition.");
    }

    #pragma omp parallel for
    for (size_t i = 0; i < v1.size(); ++i) {
        double sum = v1[i] + v2[i];
        v1[i] = sum;
        v2[i] = sum;
    }
}

double dotProduct(const std::vector<double>& v1, const std::vector<double>& v2) {
    if (v1.size() != v2.size()) {
        throw std::invalid_argument("Vectors must have the same size for dot product.");
    }
    
    double result = 0.0;
    #pragma omp parallel for reduction(+:result)
    for (size_t i = 0; i < v1.size(); ++i) {
        result += v1[i] * v2[i];
    }
    return result;
}

void calculate_degrees_and_edges_adj_parallel(
    const adj_dict_factory& adj,
    const std::vector<int>& membership,
    bool directed,
    int num_communities,
    double& e,
    double& m,
    std::vector<double>& k_out,
    std::vector<double>& k_in
    ) 
{
    const int N = membership.size();
    
    #pragma omp parallel
    {
        double local_e = 0.0;
        double local_m = 0.0;
        std::vector<double> local_k_out(num_communities, 0.0);
        std::vector<double> local_k_in(num_communities, 0.0);
        double directed_factor = directed ? 1.0 : 2.0;
        
        #pragma omp for
        for (int i = 0; i < N; i++) {
            node_t u = i + 1;
            int c1 = membership[i];
            
            auto adj_it = adj.find(u);
            if (adj_it == adj.end()) continue;
            auto& u_neighbors = adj_it->second;
            
            for (auto& v_pair : u_neighbors) {
                node_t v = v_pair.first;
                
                if (!directed && u > v) continue;
                
                int c2 = membership[v - 1];
                
                double w = 1.0;
                if (!v_pair.second.empty()) {
                    auto it = v_pair.second.begin();
                    w = it->second;
                }
                
                if (c1 == c2) {
                    local_e += directed_factor * w;
                }
                
                local_k_out[c1] += w;
                local_k_in[c2] += w;
                local_m += w;
            }
        }
        
        #pragma omp critical
        {
            e += local_e;
            m += local_m;
            for (int i = 0; i < num_communities; i++) {
                k_out[i] += local_k_out[i];
                k_in[i] += local_k_in[i];
            }
        }
    }
}

    // The input `communities` may be either:
    //   (a) a membership list: a flat sequence of ints, membership[i] = community id of node (i+1); or
    //   (b) a community list: a sequence of iterables of node ids

py::object cpp_modularity(py::object G, py::object communities, py::object weight=py::str("weight")) {
    Graph& G_ = G.cast<Graph&>();
    bool directed = G.attr("is_directed")().cast<bool>();
    adj_dict_factory& adj = G_.adj;
    const int N = G_.node.size();


    {
        bool is_empty_seq = false;
        try {
            py::sequence seq = communities.cast<py::sequence>();
            is_empty_seq = (seq.size() == 0);
        } catch (const py::cast_error&) {
            is_empty_seq = false;
        }
        if (is_empty_seq) {
            py::module warnings = py::module::import("warnings");
            warnings.attr("warn")(
                "cpp_modularity: received an empty community list; returning Q = 0.0."
            );
            return py::float_(0.0);
        }
    }


    std::vector<int> membership_vec;

    bool is_membership = false;
    py::sequence outer_seq;
    try {
        outer_seq = communities.cast<py::sequence>();
        if (outer_seq.size() > 0) {
            try {
                outer_seq[0].cast<int>();
                is_membership = true;
            } catch (const py::cast_error&) {
                is_membership = false;
            }
        }
    } catch (const py::cast_error&) {
        is_membership = true;
    }

    if (is_membership) {
        // Already a membership vector: membership[i] is the community id of the (i+1)-th node.
        membership_vec = communities.cast<std::vector<int>>();
    } else {
        membership_vec = std::vector<int>(N, -1);

        int comm_id = 0;
        for (auto community_handle : py::iter(communities)) {
            for (auto node_handle : py::iter(community_handle)) {
                py::object node = py::reinterpret_borrow<py::object>(node_handle.ptr());
                if (G_.node_to_id.contains(node)) {
                    int node_id = G_.node_to_id[node].cast<node_t>() - 1;
                    if (node_id >= 0 && node_id < N) {
                        membership_vec[node_id] = comm_id;
                    }
                }
            }
            comm_id++;
        }
    }

    int num_communities = membership_vec.size();
    
    double e = 0.0;
    double m = 0.0;
    std::vector<double> k_out(num_communities, 0.0);
    std::vector<double> k_in(num_communities, 0.0);
    
    calculate_degrees_and_edges_adj_parallel(adj, membership_vec, directed, num_communities, e, m, k_out, k_in);
    
    if (!directed) addVectorsInPlace(k_out, k_in);

    // Handle empty graph / zero total edge weight: m == 0 makes
    // `norm = 1.0 / (directed_factor * m)` divide by zero and produces
    // inf / nan. Define Q = 0.0 in this degenerate case (no edges -> no
    // community structure to measure), with a Python warning.
    if (m == 0.0) {
        py::module warnings = py::module::import("warnings");
        warnings.attr("warn")(
            "cpp_modularity: graph has no edges (m == 0); returning Q = 0.0."
        );
        return py::float_(0.0);
    }

    double directed_factor = directed ? 1.0 : 2.0;
    double norm = 1.0 / (directed_factor * m);
    e *= norm;
    
    double sum_products = dotProduct(k_out, k_in);
    sum_products *= norm * norm;
    
    double Q = e - sum_products;
    
    return py::float_(Q);
}