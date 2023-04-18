#include "pagerank.h"
#include "../../classes/directed_graph.h"
#include "../../common/utils.h"
#include "../../classes/linkgraph.h"
#include "time.h"

struct Page {
	Page(){} 
    Page(const double &_newPR, const double &_oldPR) {newPR = _newPR; oldPR = _oldPR;}

    double newPR, oldPR;
};

// outDegree
// get_edge_from_node

py::object _pagerank(py::object G, double alpha=0.85, int max_iterator=500, double threshold=1e-6) {

    bool is_directed = G.attr("is_directed")().cast<bool>();
    if (is_directed == false) {
        printf("PageRank is designed for directed graphs.\n");
        return py::dict();
    }
    DiGraph& G_ = G.cast<DiGraph&>();
    int N = G_.node.size();
    // Graph_L G_l = graph_to_linkgraph(G_, is_directed, "", true);
    Graph_L G_l;
    if(G_.linkgraph_dirty){
        G_l = graph_to_linkgraph(G_, is_directed, "", true);
        G_.linkgraph_structure=G_l;
        G_.linkgraph_dirty = false;
    }
    else{
        G_l = G_.linkgraph_structure;
    }

    std::vector<LinkEdge>& E = G_l.edges;
    std::vector<int> outDegree = G_l.degree;
    std::vector<int> head = G_l.head;

    std::vector<Page>page(N+1);
    for (int i = 1; i < N + 1; ++i) {
        page[i] = Page(0, 1.0/N);
    }

    int cnt = 0; //统计迭代几轮
	int shouldStop = 0; //根据oldPR与newPR的差值 判断是否停止迭代


    while(!shouldStop)
    {
        shouldStop = 1;
        double res = 0;
        for(int i = 1; i < N+1; ++i) {
            if (outDegree[i] == 0) {
                res += page[i].oldPR;
                continue;
            }
            double tmpPR = (page[i].oldPR / outDegree[i]) * alpha;
            for(int p = head[i]; p != -1; p = E[p].next){
                page[E[p].to].newPR += tmpPR;
            }
        }
        double sum = 0;
        for(int i = 1; i < N+1; ++i)
        {
            page[i].newPR += (1 - alpha) / N + res / N * alpha;
            sum += fabs(page[i].newPR - page[i].oldPR);

            page[i].oldPR = page[i].newPR;
            page[i].newPR = 0;
        }
        
        if (sum > threshold * N)
            shouldStop = 0;
        cnt++;
        if (cnt >= max_iterator)
            break;
    }
    
    py::list res_lst = py::list();
    for(int i = 1;i < N + 1;i++){
        res_lst.append(page[i].oldPR);
    }

    return res_lst;
}
