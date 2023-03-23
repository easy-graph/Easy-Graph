#include "k_cores.h"
#include "../../classes/graph.h"
#include "../../common/utils.h"
#include "../../classes/linkgraph.h"
#include<time.h>

py::object core_decomposition(py::object G) {
    // reference:https://arxiv.org/pdf/cs/0310049.pdf
    // clock_t start_time = clock();
    Graph& G_ = G.cast<Graph&>();
    int N = G_.node.size();
    bool is_directed = G.attr("is_directed")().cast<bool>();
    // Graph_L G_l = graph_to_linkgraph(G_, is_directed,"", true);
    clock_t start_time = clock();
    const Graph_L& G_l = graph_to_linkgraph(G_, is_directed, "", true, false);
    clock_t end_time = clock();
    printf("cost0:%2f\n",(double)(end_time-start_time)/CLOCKS_PER_SEC);

    std::vector<LinkEdge> edges = G_l.edges;
    int edges_num = edges.size();
    std::vector<int> deg = G_l.degree;
    std::vector<int> head = G_l.head;
    int max_deg = G_l.max_deg;
    // bin[i] indicates how many nodes of degree i there are

    std::vector<int> core(N+1, 0);
    std::vector<int> bin(N+1, 0);
    std::vector<int> pos(N+1, 0);
    std::vector<int> vert(N+1, 0);
    start_time = clock();
    // 基数排序
    for(int i = 1; i <= N; ++i)
        ++bin[deg[i]];
    end_time = clock();
    printf("cost1:%2f\n",(double)(end_time-start_time)/CLOCKS_PER_SEC);
     //此时bin[i]表示度数为i的点有多少个
    int start = 1;
    start_time = clock();
    for(int i = 0; i <= max_deg; ++i){
        int num = bin[i];
        bin[i] = start;
        start += num;
    }
    end_time = clock();
    printf("cost2:%2f\n",(double)(end_time-start_time)/CLOCKS_PER_SEC);

    start_time = clock();
    for(int i = 1; i <= N; ++i){
        pos[i] = bin[deg[i]];
        vert[pos[i]] = i;
        ++bin[deg[i]];
    }
    end_time = clock();
    printf("cost3:%2f\n",(double)(end_time-start_time)/CLOCKS_PER_SEC);

    start_time = clock();
    //此时bin[i]表示度数为i+1的第一个点，所以后面要进行左移操作
    for(int i = max_deg; i >= 1; --i){
        bin[i] = bin[i - 1];
    }
    end_time = clock();
    printf("cost4:%2f\n",(double)(end_time-start_time)/CLOCKS_PER_SEC);
    bin[0] = 1;
    start_time = clock();
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
    end_time = clock();
    printf("cost5:%2f\n",(double)(end_time-start_time)/CLOCKS_PER_SEC);
    // clock_t end_time = clock();
    // printf("cost:%2f\n",(double)(end_time-start_time)/CLOCKS_PER_SEC);
    start_time = clock();
    py::list core_list = py::list();
    for(register int i = 1; i <= N; ++i){
        core_list.append(core[i]);
    }
    end_time = clock();
    printf("cost6:%2f\n",(double)(end_time-start_time)/CLOCKS_PER_SEC);
    return core_list;
}
