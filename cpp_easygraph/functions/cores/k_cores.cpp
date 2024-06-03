#ifdef EASYGRAPH_ENABLE_GPU
#include <gpu_easygraph.h>
#endif

#include "k_cores.h"
#include "../../classes/graph.h"
#include "../../common/utils.h"
#include "../../classes/linkgraph.h"
#include <time.h>


py::object invoke_cpp_core_decomposition(py::object G) {
    // reference:https://arxiv.org/pdf/cs/0310049.pdf
    Graph& G_ = G.cast<Graph&>();
    int N = G_.node.size();
    bool is_directed = G.attr("is_directed")().cast<bool>();
    Graph_L G_l;
    if(G_.linkgraph_dirty || G_.linkgraph_structure.max_deg == -1){
        G_l = graph_to_linkgraph(G_, is_directed, "", true, false);
        G_.linkgraph_dirty = false;
    }
    else{
        G_l = G_.linkgraph_structure;
    }
    std::vector<LinkEdge> edges = G_l.edges;
    int edges_num = edges.size();
    std::vector<int> deg = G_l.degree;
    std::vector<int> head = G_l.head;
    int max_deg = G_l.max_deg;
    std::vector<int> core(N+1, 0);
    std::vector<int> bin(max_deg+1, 0);
    std::vector<int> pos(N+1, 0);
    std::vector<int> vert(N+1, 0);
    for(int i = 1; i <= N; ++i)
        ++bin[deg[i]];
    int start = 1;
    for(int i = 0; i <= max_deg; ++i){
        int num = bin[i];
        bin[i] = start;
        start += num;
    }
    for(int i = 1; i <= N; ++i){
        pos[i] = bin[deg[i]];
        vert[pos[i]] = i;
        ++bin[deg[i]];
    }
    for(int i = max_deg; i >= 1; --i){
        bin[i] = bin[i - 1];
    }
    bin[0] = 1;
    for(int i = 1; i <= N; ++i){
        int v = vert[i];
        core[v] = deg[v];
        for(register int p = head[v]; p!=-1; p = edges[p].next){
            int u = edges[p].to;
            if (deg[u] > deg[v]) {
                int w = vert[bin[deg[u]]];
                if(u != w){
                    std::swap(vert[pos[u]],vert[pos[w]]);
                    std::swap(pos[u],pos[w]);
                }
                ++bin[deg[u]];
                --deg[u];
            }
        }
    }

    py::array::ShapeContainer ret_shape{(int)core.size()};
    py::array_t<int> ret(ret_shape, core.data());

    return ret;
}

#ifdef EASYGRAPH_ENABLE_GPU
py::object invoke_gpu_core_decomposition(py::object G) {
    Graph& G_ = G.cast<Graph&>();
    auto csr_graph = G_.gen_CSR();
    std::vector<int>& E = csr_graph->E;
    std::vector<int>& V = csr_graph->V;
    std::vector<int> KC;
    int gpu_r = gpu_easygraph::k_core(V, E, KC);

    if (gpu_r != gpu_easygraph::EG_GPU_SUCC) {
        // the code below will throw an exception
        py::pybind11_fail(gpu_easygraph::err_code_detail(gpu_r));
    }

    py::array::ShapeContainer ret_shape{(int)KC.size()};
    py::array_t<int> ret(ret_shape, KC.data());

    return ret;
}
#endif

py::object core_decomposition(py::object G) {
#ifdef EASYGRAPH_ENABLE_GPU
    return invoke_gpu_core_decomposition(G);
#else
    return invoke_cpp_core_decomposition(G);
#endif
}