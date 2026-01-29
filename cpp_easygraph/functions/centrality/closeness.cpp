#include "centrality.h"

#ifdef EASYGRAPH_ENABLE_GPU
#include <gpu_easygraph.h>
#endif

#include "../../classes/graph.h"
#include "../../common/utils.h"
#include "../../classes/linkgraph.h"

#include <queue>
#include <vector>
#include <limits>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

// Heap node: use negative value + max heap to implement min heap
typedef std::pair<float, int> HeapNode;

// Optimized adjacency list cache
struct FastAdjCache {
    std::vector<int*> neighbor_ptrs;
    std::vector<int> neighbor_counts;
    std::vector<float*> weight_ptrs;
    std::vector<std::vector<int>> neighbor_storage;
    std::vector<std::vector<float>> weight_storage;
    
    void init(int N) {
        neighbor_ptrs.resize(N + 1, nullptr);
        neighbor_counts.resize(N + 1, 0);
        neighbor_storage.resize(N + 1);
    }
    
    void init_with_weights(int N) {
        init(N);
        weight_ptrs.resize(N + 1, nullptr);
        weight_storage.resize(N + 1);
    }
    
    inline void build_if_needed(int u, const std::vector<int>& head, 
                                const std::vector<LinkEdge>& edges, bool store_weights) {
        if (neighbor_ptrs[u] != nullptr) return;
        
        std::vector<int>& neis = neighbor_storage[u];
        for (int p = head[u]; p != -1; p = edges[p].next) {
            neis.push_back(edges[p].to);
            if (store_weights) {
                weight_storage[u].push_back(edges[p].w);
            }
        }
        
        neighbor_counts[u] = neis.size();
        neighbor_ptrs[u] = neis.data();
        if (store_weights) {
            weight_ptrs[u] = weight_storage[u].data();
        }
    }
    
    inline int* get_neighbors_ptr(int u) const { return neighbor_ptrs[u]; }
    inline int get_neighbor_count(int u) const { return neighbor_counts[u]; }
    inline float* get_weights_ptr(int u) const { return weight_ptrs[u]; }
};

// BFS implementation - directly use raw adjacency list
double closeness_bfs_direct(const Graph_L& G_l, const int &S, int cutoff,
                            std::vector<int>& already_counted,
                            std::vector<int>& queue_storage,
                            int timestamp) {
    int N = G_l.n;
    const std::vector<LinkEdge>& E = G_l.edges;
    const std::vector<int>& head = G_l.head;
    
    int nodes_reached = 0;
    long long sum_dis = 0;
    
    queue_storage.clear();
    
    int queue_front = 0;
    already_counted[S] = timestamp;
    queue_storage.push_back(S);
    queue_storage.push_back(0);
    
    while (queue_front < static_cast<int>(queue_storage.size())) {
        int u = queue_storage[queue_front++];
        int actdist = queue_storage[queue_front++];
        
        if (cutoff >= 0 && actdist > cutoff) {
            continue;
        }
        
        sum_dis += actdist;
        nodes_reached++;
        
        for (int p = head[u]; p != -1; p = E[p].next) {
            int v = E[p].to;
            
            if (already_counted[v] == timestamp) {
                continue;
            }
            
            already_counted[v] = timestamp;
            queue_storage.push_back(v);
            queue_storage.push_back(actdist + 1);
        }
    }
    
    if (nodes_reached == 1)
        return 0.0;
    else
        return 1.0 * (nodes_reached - 1) * (nodes_reached - 1) / ((N - 1) * sum_dis);
}

// Check if the graph is unweighted
inline bool is_unweighted_graph(const Graph_L& G_l) {
    const std::vector<LinkEdge>& E = G_l.edges;
    for (const auto& edge : E) {
        if (std::abs(edge.w - 1.0f) > 1e-9) {
            return false;
        }
    }
    return true;
}

// Dijkstra implementation - use on-demand adjacency cache
double closeness_dijkstra_cached(const Graph_L& G_l, const int &S, int cutoff,
                                 std::vector<float>& dist,
                                 std::vector<int>& which,
                                 FastAdjCache& cache,
                                 int timestamp) {
    int N = G_l.n;
    const std::vector<LinkEdge>& E = G_l.edges;
    const std::vector<int>& head = G_l.head;
    
    int nodes_reached = 0;
    double sum_dis = 0.0;
    
    std::priority_queue<HeapNode> heap;
    
    dist[S] = 1.0f;
    which[S] = timestamp;
    heap.push({-1.0f, S});
    
    while (!heap.empty()) {
        HeapNode top = heap.top();
        heap.pop();
        float mindist = -top.first;
        int minnei = top.second;
        
        if (mindist > dist[minnei]) {
            continue;
        }
        
        float actual_dist = mindist - 1.0f;
        if (cutoff >= 0 && actual_dist > cutoff) {
            continue;
        }
        
        sum_dis += actual_dist;
        nodes_reached++;
        
        cache.build_if_needed(minnei, head, E, true);
        
        int* neis = cache.get_neighbors_ptr(minnei);
        float* ws = cache.get_weights_ptr(minnei);
        int nlen = cache.get_neighbor_count(minnei);
        
        for (int j = 0; j < nlen; j++) {
            int to = neis[j];
            float altdist = mindist + ws[j];
            float curdist = dist[to];
            
            if (which[to] != timestamp) {
                which[to] = timestamp;
                dist[to] = altdist;
                heap.push({-altdist, to});
            } else if (curdist == 0.0f || altdist < curdist) {
                dist[to] = altdist;
                heap.push({-altdist, to});
            }
        }
    }
    
    if (nodes_reached == 1)
        return 0.0;
    else
        return 1.0 * (nodes_reached - 1) * (nodes_reached - 1) / ((N - 1) * sum_dis);
}

static py::object invoke_cpp_closeness_centrality(py::object G, py::object weight,
                                            py::object cutoff, py::object sources) {
    Graph& G_ = G.cast<Graph&>();
    int N = G_.node.size();
    bool is_directed = G.attr("is_directed")().cast<bool>();
    std::string weight_key = weight_to_string(weight);
    const Graph_L& G_l = graph_to_linkgraph(G_, is_directed, weight_key, false, false);
    int cutoff_ = -1;
    if (!cutoff.is_none()){
        cutoff_ = cutoff.cast<int>();
    }
    
    // Auto algorithm selection
    bool use_bfs = (weight.is_none() || is_unweighted_graph(G_l));
    
    std::vector<double> CC;
    
    if(!sources.is_none()){
        py::list sources_list = py::list(sources);
        int sources_list_len = py::len(sources_list);
        CC.resize(sources_list_len);
        
        // Collect all source node IDs
        std::vector<node_t> source_ids(sources_list_len);
        for(int i = 0; i < sources_list_len; i++){
            if(G_.node_to_id.attr("get")(sources_list[i],py::none()).is_none()){
                printf("The node should exist in the graph!");
                return py::none();
            }
            source_ids[i] = G_.node_to_id.attr("get")(sources_list[i]).cast<node_t>();
        }
        
        // OpenMP parallel computation
        // Only enable parallelism when sources are many to avoid overhead
        #pragma omp parallel if(sources_list_len > 100)
        {
            // Per-thread data structures (avoid race conditions)
            std::vector<int> already_counted(N + 1, 0);
            std::vector<int> queue_storage;
            queue_storage.reserve(N * 2);
            
            std::vector<float> dist(N + 1, 0.0f);
            std::vector<int> which(N + 1, 0);
            
            FastAdjCache cache;
            if (!use_bfs) {
                cache.init_with_weights(N);
            }
            
            // Assign unique timestamp start for each thread
            #ifdef _OPENMP
            int thread_id = omp_get_thread_num();
            int num_threads = omp_get_num_threads();
            int timestamp = thread_id * sources_list_len;
            #else
            int timestamp = 0;
            #endif
            
            // Parallel loop: each thread handles different source node
            #pragma omp for schedule(dynamic, 1)
            for(int i = 0; i < sources_list_len; i++){
                timestamp++;
                double res;
                if (use_bfs) {
                    res = closeness_bfs_direct(G_l, source_ids[i], cutoff_, 
                                              already_counted, queue_storage, timestamp);
                } else {
                    res = closeness_dijkstra_cached(G_l, source_ids[i], cutoff_, 
                                                   dist, which, cache, timestamp);
                }
                CC[i] = res;
            }
        }
    }
    else{
        CC.resize(N);
        
        // OpenMP parallel computation for all nodes
        // Only enable parallelism when node count is large
        #pragma omp parallel if(N > 100)
        {
            // Per-thread data structures
            std::vector<int> already_counted(N + 1, 0);
            std::vector<int> queue_storage;
            queue_storage.reserve(N * 2);
            
            std::vector<float> dist(N + 1, 0.0f);
            std::vector<int> which(N + 1, 0);
            
            FastAdjCache cache;
            if (!use_bfs) {
                cache.init_with_weights(N);
            }
            
            // Assign unique timestamp start for each thread
            #ifdef _OPENMP
            int thread_id = omp_get_thread_num();
            int num_threads = omp_get_num_threads();
            int timestamp = thread_id * N;
            #else
            int timestamp = 0;
            #endif
            
            // Parallel loop: dynamic scheduling for load balancing
            #pragma omp for schedule(dynamic, 10)
            for(int i = 1; i <= N; i++){
                timestamp++;
                double res;
                if (use_bfs) {
                    res = closeness_bfs_direct(G_l, i, cutoff_, 
                                              already_counted, queue_storage, timestamp);
                } else {
                    res = closeness_dijkstra_cached(G_l, i, cutoff_, 
                                                   dist, which, cache, timestamp);
                }
                CC[i - 1] = res;
            }
        }
    }
    
    py::array::ShapeContainer ret_shape{(int)CC.size()};
    py::array_t<double> ret(ret_shape, CC.data());

    return ret;
}

#ifdef EASYGRAPH_ENABLE_GPU
static py::object invoke_gpu_closeness_centrality(py::object G, py::object weight, py::object py_sources) {
    Graph& G_ = G.cast<Graph&>();
    if (weight.is_none()) {
        G_.gen_CSR();
    } else {
        G_.gen_CSR(weight_to_string(weight));
    }
    auto csr_graph = G_.csr_graph;
    std::vector<int>& E = csr_graph->E;
    std::vector<int>& V = csr_graph->V;
    std::vector<double> *W_p = weight.is_none() ? &(csr_graph->unweighted_W)
                                : csr_graph->W_map.find(weight_to_string(weight))->second.get();
    auto sources = G_.gen_CSR_sources(py_sources);
    std::vector<double> CC;
    int gpu_r = gpu_easygraph::closeness_centrality(V, E, *W_p, *sources, CC);

    if (gpu_r != gpu_easygraph::EG_GPU_SUCC) {
        // the code below will throw an exception
        py::pybind11_fail(gpu_easygraph::err_code_detail(gpu_r));
    }

    py::array::ShapeContainer ret_shape{(int)CC.size()};
    py::array_t<double> ret(ret_shape, CC.data());

    return ret;
}
#endif

py::object closeness_centrality(py::object G, py::object weight, py::object cutoff, py::object sources) {
#ifdef EASYGRAPH_ENABLE_GPU
    return invoke_gpu_closeness_centrality(G, weight, sources);
#else
    return invoke_cpp_closeness_centrality(G, weight, cutoff, sources);
#endif
}