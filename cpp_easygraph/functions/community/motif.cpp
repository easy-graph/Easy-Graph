#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <random>
#include <algorithm>
#include <queue>
#include <bitset>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <omp.h>
#include "../../classes/graph.h"
#include "../../classes/linkgraph.h"
#include "motif.h"

namespace py = pybind11;
using namespace std;

const int MAX_NODES = 65536;

struct MotifResult {
    vector<int> nodes;
};

class MotifEnumerator {
private:
    Graph_L GL;
    int k;
    int v_start;
    vector<int>& id_to_node;
    mt19937 rng;
    uniform_real_distribution<double> dist;
    vector<double> cut_prob;
    
    vector<int> temp_nexcl;
    vector<MotifResult> results;
    bitset<MAX_NODES> nvp_bitset;

public:
    vector<int> node_list;
    vector<int> neighbor_offsets;
    vector<int> neighbor_data;
    int n_edges;

    void build_neighbor_structure() {
        neighbor_offsets.resize(GL.n + 2, 0);
        for (int i = 1; i <= GL.n; ++i) {
            int count = 0;
            for (int e = GL.head[i]; e != -1; e = GL.edges[e].next) {
                count++;
            }
            neighbor_offsets[i + 1] = neighbor_offsets[i] + count;
        }
        
        n_edges = neighbor_offsets[GL.n + 1];
        neighbor_data.resize(n_edges);
        
        for (int i = 1; i <= GL.n; ++i) {
            int idx = neighbor_offsets[i];
            for (int e = GL.head[i]; e != -1; e = GL.edges[e].next) {
                neighbor_data[idx++] = GL.edges[e].to;
            }
        }

        node_list.reserve(GL.n);
        for (int i = 1; i <= GL.n; ++i) {
            if (GL.head[i] != -1) {
                node_list.push_back(i);
            }
        }
    }

    int get_neighbor_count(int v) const {
        return neighbor_offsets[v + 1] - neighbor_offsets[v];
    }

    const int* get_neighbors_ptr(int v) const {
        return neighbor_data.data() + neighbor_offsets[v];
    }

    void exclusive_neighborhood(int v, const vector<int>& vp, vector<int>& result) {
        result.clear();
        
        const int* nbr_ptr = get_neighbors_ptr(v);
        int nbr_count = get_neighbor_count(v);
        
        nvp_bitset.reset();
        for (int node : vp) {
            nvp_bitset.set(node);
            const int* node_nbr_ptr = get_neighbors_ptr(node);
            int node_nbr_count = get_neighbor_count(node);
            for (int i = 0; i < node_nbr_count; ++i) {
                nvp_bitset.set(node_nbr_ptr[i]);
            }
        }
        
        for (int i = 0; i < nbr_count; ++i) {
            int neighbor = nbr_ptr[i];
            if (!nvp_bitset.test(neighbor)) {
                result.push_back(neighbor);
            }
        }
    }

    void extend_subgraph_local(const vector<int>& Vsubgraph, const vector<int>& Vextension, int local_v_start, 
                              vector<int>& local_temp_nexcl, bitset<MAX_NODES>& local_nvp_bitset,
                              vector<MotifResult>& local_results) {
        if ((int)Vsubgraph.size() == k) {
            local_results.push_back({Vsubgraph});
            return;
        }

        int original_size = Vextension.size();
        for (int i = original_size - 1; i >= 0; --i) {
            int w = Vextension[i];
            
            if (!cut_prob.empty() && (int)cut_prob.size() > (int)Vsubgraph.size() && dist(rng) > cut_prob[Vsubgraph.size()]) {
                continue;
            }
            
            vector<int> new_subgraph;
            new_subgraph.reserve(Vsubgraph.size() + 1);
            new_subgraph.assign(Vsubgraph.begin(), Vsubgraph.end());
            new_subgraph.push_back(w);

            exclusive_neighborhood_local(w, Vsubgraph, local_temp_nexcl, local_nvp_bitset);

            vector<int> next_ext;
            next_ext.reserve(original_size + local_temp_nexcl.size());
            for (int j = 0; j < original_size; ++j) {
                if (j != i) {
                    int u = Vextension[j];
                    if (u > local_v_start) {
                        next_ext.push_back(u);
                    }
                }
            }
            for (int u : local_temp_nexcl) {
                if (u > local_v_start) {
                    next_ext.push_back(u);
                }
            }

            extend_subgraph_local(new_subgraph, next_ext, local_v_start, local_temp_nexcl, local_nvp_bitset, local_results);
        }
    }

    int extend_subgraph_count_local(const vector<int>& Vsubgraph, const vector<int>& Vextension, int local_v_start, vector<int>& local_temp_nexcl, bitset<MAX_NODES>& local_nvp_bitset) {
        if ((int)Vsubgraph.size() == k) {
            return 1;
        }

        int count = 0;
        int original_size = Vextension.size();
        for (int i = original_size - 1; i >= 0; --i) {
            int w = Vextension[i];
            
            vector<int> new_subgraph;
            new_subgraph.reserve(Vsubgraph.size() + 1);
            new_subgraph.assign(Vsubgraph.begin(), Vsubgraph.end());
            new_subgraph.push_back(w);

            exclusive_neighborhood_local(w, Vsubgraph, local_temp_nexcl, local_nvp_bitset);

            vector<int> next_ext;
            next_ext.reserve(original_size + local_temp_nexcl.size());
            for (int j = 0; j < original_size; ++j) {
                if (j != i) {
                    int u = Vextension[j];
                    if (u > local_v_start) {
                        next_ext.push_back(u);
                    }
                }
            }
            for (int u : local_temp_nexcl) {
                if (u > local_v_start) {
                    next_ext.push_back(u);
                }
            }

            count += extend_subgraph_count_local(new_subgraph, next_ext, local_v_start, local_temp_nexcl, local_nvp_bitset);
        }
        return count;
    }

    int extend_subgraph_count(const vector<int>& Vsubgraph, const vector<int>& Vextension) {
        return extend_subgraph_count_local(Vsubgraph, Vextension, v_start, temp_nexcl, nvp_bitset);
    }

    void exclusive_neighborhood_local(int v, const vector<int>& vp, vector<int>& result, bitset<MAX_NODES>& local_nvp_bitset) {
        result.clear();
        
        const int* nbr_ptr = get_neighbors_ptr(v);
        int nbr_count = get_neighbor_count(v);
        
        local_nvp_bitset.reset();
        for (int node : vp) {
            local_nvp_bitset.set(node);
            const int* node_nbr_ptr = get_neighbors_ptr(node);
            int node_nbr_count = get_neighbor_count(node);
            for (int i = 0; i < node_nbr_count; ++i) {
                local_nvp_bitset.set(node_nbr_ptr[i]);
            }
        }
        
        for (int i = 0; i < nbr_count; ++i) {
            int neighbor = nbr_ptr[i];
            if (!local_nvp_bitset.test(neighbor)) {
                result.push_back(neighbor);
            }
        }
    }

public:
    MotifEnumerator(Graph_L gl, int k, vector<int>& id_to_node)
        : GL(gl), k(k), v_start(0), id_to_node(id_to_node), rng(42), dist(0.0, 1.0) {
        build_neighbor_structure();
        temp_nexcl.reserve(gl.n);
    }

    MotifEnumerator(Graph_L gl, int k, vector<int>& id_to_node, const vector<double>& cut_prob_vec, int seed)
        : GL(gl), k(k), v_start(0), id_to_node(id_to_node), cut_prob(cut_prob_vec), rng(seed), dist(0.0, 1.0) {
        build_neighbor_structure();
        temp_nexcl.reserve(gl.n);
    }

    py::array_t<int> enumerate() {
        results.clear();
        
        if (k == 2) {
            for (int v : node_list) {
                if (!cut_prob.empty() && dist(rng) > cut_prob[0]) {
                    continue;
                }
                const int* nbr_ptr = get_neighbors_ptr(v);
                int nbr_count = get_neighbor_count(v);
                for (int j = 0; j < nbr_count; ++j) {
                    int u = nbr_ptr[j];
                    if (u > v) {
                        results.push_back({vector<int>{v, u}});
                    }
                }
            }
            return convert_results_to_numpy();
        }
        
        vector<vector<MotifResult>> thread_results(8);
        
        omp_set_num_threads(8);
        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            vector<MotifResult>& local_results = thread_results[thread_id];
            local_results.reserve(10000);
            
            #pragma omp for schedule(dynamic)
            for (int i = 0; i < (int)node_list.size(); ++i) {
                int v = node_list[i];
                
                if (!cut_prob.empty() && dist(rng) > cut_prob[0]) {
                    continue;
                }
                
                int local_v_start = v;
                vector<int> Vsubgraph;
                Vsubgraph.reserve(k);
                Vsubgraph.push_back(v);

                vector<int> Vextension;
                Vextension.reserve(GL.n);
                const int* nbr_ptr = get_neighbors_ptr(v);
                int nbr_count = get_neighbor_count(v);
                for (int j = 0; j < nbr_count; ++j) {
                    int u = nbr_ptr[j];
                    if (u > v) {
                        Vextension.push_back(u);
                    }
                }

                vector<int> local_temp_nexcl;
                local_temp_nexcl.reserve(GL.n);
                bitset<MAX_NODES> local_nvp_bitset;
                
                extend_subgraph_local(Vsubgraph, Vextension, local_v_start, local_temp_nexcl, local_nvp_bitset, local_results);
            }
        }
        
        for (const auto& tr : thread_results) {
            results.insert(results.end(), tr.begin(), tr.end());
        }
        
        return convert_results_to_numpy();
    }

    void extend_subgraph(const vector<int>& Vsubgraph, const vector<int>& Vextension) {
        if ((int)Vsubgraph.size() == k) {
            results.push_back({Vsubgraph});
            return;
        }

        int original_size = Vextension.size();
        for (int i = original_size - 1; i >= 0; --i) {
            int w = Vextension[i];
            
            if (w <= v_start) {
                continue;
            }
            
            vector<int> new_subgraph;
            new_subgraph.reserve(Vsubgraph.size() + 1);
            new_subgraph.assign(Vsubgraph.begin(), Vsubgraph.end());
            new_subgraph.push_back(w);

            exclusive_neighborhood(w, Vsubgraph, temp_nexcl);

            vector<int> next_ext;
            next_ext.reserve(original_size + temp_nexcl.size());
            for (int j = 0; j < original_size; ++j) {
                if (j != i) {
                    int u = Vextension[j];
                    if (u > v_start) {
                        next_ext.push_back(u);
                    }
                }
            }
            for (int u : temp_nexcl) {
                if (u > v_start) {
                    next_ext.push_back(u);
                }
            }

            extend_subgraph(new_subgraph, next_ext);
        }
    }

    int count_enumerate() {
        int count = 0;
        
        if (k == 2) {
            for (int v : node_list) {
                const int* nbr_ptr = get_neighbors_ptr(v);
                int nbr_count = get_neighbor_count(v);
                for (int j = 0; j < nbr_count; ++j) {
                    int u = nbr_ptr[j];
                    if (u > v) {
                        count++;
                    }
                }
            }
            return count;
        }
        
        omp_set_num_threads(8);
        #pragma omp parallel reduction(+:count)
        {
            #pragma omp for schedule(dynamic)
            for (int i = 0; i < (int)node_list.size(); ++i) {
                int v = node_list[i];
                int local_v_start = v;
                vector<int> Vsubgraph;
                Vsubgraph.reserve(k);
                Vsubgraph.push_back(v);

                vector<int> Vextension;
                Vextension.reserve(GL.n);
                const int* nbr_ptr = get_neighbors_ptr(v);
                int nbr_count = get_neighbor_count(v);
                for (int j = 0; j < nbr_count; ++j) {
                    int u = nbr_ptr[j];
                    if (u > v) {
                        Vextension.push_back(u);
                    }
                }

                vector<int> local_temp_nexcl;
                local_temp_nexcl.reserve(GL.n);
                bitset<MAX_NODES> local_nvp_bitset;

                count += extend_subgraph_count_local(Vsubgraph, Vextension, local_v_start, local_temp_nexcl, local_nvp_bitset);
            }
        }
        
        return count;
    }

    py::list convert_results_to_python() {
        py::list py_results;
        for (const auto& result : results) {
            py::set py_set;
            for (int node_id : result.nodes) {
                if (node_id > 0 && node_id < (int)id_to_node.size()) {
                    py_set.add(py::cast(id_to_node[node_id]));
                }
            }
            py_results.append(py_set);
        }
        return py_results;
    }

    py::array_t<int> convert_results_to_numpy() {
        size_t num_results = results.size();
        size_t k_size = k;
        
        int* data = new int[num_results * k_size];
        
        size_t idx = 0;
        for (const auto& result : results) {
            for (size_t i = 0; i < k_size; ++i) {
                if (i < result.nodes.size() && result.nodes[i] > 0 && result.nodes[i] < (int)id_to_node.size()) {
                    data[idx++] = id_to_node[result.nodes[i]];
                } else {
                    data[idx++] = -1;
                }
            }
        }
        
        // Create a capsule to manage the memory
        py::capsule capsule(data, [](void* ptr) {
            delete[] static_cast<int*>(ptr);
        });
        
        // Create numpy array with the capsule
        py::array_t<int> result_array = py::array_t<int>(
            {num_results, k_size},  // shape
            {k_size * sizeof(int), sizeof(int)},  // strides
            data,
            capsule
        );
        
        return result_array;
    }

    const vector<MotifResult>& get_results() const {
        return results;
    }
    
    size_t get_k() const {
        return k;
    }
};

py::list cpp_enumerate_subgraph(py::object G, int k) {
    Graph& G_ = G.cast<Graph&>();
    Graph_L GL = G_.linkgraph_structure;
    
    py::dict id_to_node_py = G_.id_to_node;
    vector<int> id_to_node_vec;
    id_to_node_vec.reserve(id_to_node_py.size() + 1);
    id_to_node_vec.push_back(0);
    for (int i = 1; i <= (int)id_to_node_py.size(); ++i) {
        py::object node_obj = id_to_node_py[py::cast(i)];
        id_to_node_vec.push_back(node_obj.cast<int>());
    }
    
    MotifEnumerator enumerator(GL, k, id_to_node_vec);
    enumerator.enumerate();
    return enumerator.convert_results_to_python();
}

py::int_ cpp_count_enumerate_subgraph(py::object G, int k) {
    Graph& G_ = G.cast<Graph&>();
    Graph_L GL = G_.linkgraph_structure;
    
    py::dict id_to_node_py = G_.id_to_node;
    vector<int> id_to_node_vec;
    id_to_node_vec.reserve(id_to_node_py.size() + 1);
    id_to_node_vec.push_back(0);
    for (int i = 1; i <= (int)id_to_node_py.size(); ++i) {
        py::object node_obj = id_to_node_py[py::cast(i)];
        id_to_node_vec.push_back(node_obj.cast<int>());
    }
    
    MotifEnumerator enumerator(GL, k, id_to_node_vec);
    int count = enumerator.count_enumerate();
    return count;
}

py::list cpp_random_enumerate_subgraph(py::object G, int k, py::object cut_prob) {
    Graph& G_ = G.cast<Graph&>();
    Graph_L GL = G_.linkgraph_structure;
    
    py::dict id_to_node_py = G_.id_to_node;
    vector<int> id_to_node_vec;
    id_to_node_vec.reserve(id_to_node_py.size() + 1);
    id_to_node_vec.push_back(0);
    for (int i = 1; i <= (int)id_to_node_py.size(); ++i) {
        py::object node_obj = id_to_node_py[py::cast(i)];
        id_to_node_vec.push_back(node_obj.cast<int>());
    }

    vector<double> cut_prob_vec;
    if (py::isinstance<py::list>(cut_prob)) {
        py::list cut_prob_list = cut_prob.cast<py::list>();
        cut_prob_vec.reserve(cut_prob_list.size());
        for (size_t i = 0; i < cut_prob_list.size(); ++i) {
            cut_prob_vec.push_back(cut_prob_list[i].cast<double>());
        }
    } else {
        throw std::runtime_error("cut_prob must be a list");
    }

    if (cut_prob_vec.size() != k) {
        throw py::value_error("length of cut_prob invalid, should equal to k");
    }

    MotifEnumerator enumerator(GL, k, id_to_node_vec, cut_prob_vec, 42);
    enumerator.enumerate();
    return enumerator.convert_results_to_python();
}

py::list cpp_random_enumerate_subgraph_with_seed(py::object G, int k, py::object cut_prob, int seed) {
    Graph& G_ = G.cast<Graph&>();
    Graph_L GL = G_.linkgraph_structure;
    
    py::dict id_to_node_py = G_.id_to_node;
    vector<int> id_to_node_vec;
    id_to_node_vec.reserve(id_to_node_py.size() + 1);
    id_to_node_vec.push_back(0);
    for (int i = 1; i <= (int)id_to_node_py.size(); ++i) {
        py::object node_obj = id_to_node_py[py::cast(i)];
        id_to_node_vec.push_back(node_obj.cast<int>());
    }

    vector<double> cut_prob_vec;
    if (py::isinstance<py::list>(cut_prob)) {
        py::list cut_prob_list = cut_prob.cast<py::list>();
        cut_prob_vec.reserve(cut_prob_list.size());
        for (size_t i = 0; i < cut_prob_list.size(); ++i) {
            cut_prob_vec.push_back(cut_prob_list[i].cast<double>());
        }
    } else {
        throw std::runtime_error("cut_prob must be a list");
    }

    if (cut_prob_vec.size() != k) {
        throw py::value_error("length of cut_prob invalid, should equal to k");
    }

    MotifEnumerator enumerator(GL, k, id_to_node_vec, cut_prob_vec, seed);
    enumerator.enumerate();
    return enumerator.convert_results_to_python();
}
