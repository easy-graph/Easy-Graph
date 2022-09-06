#include "mst.h"
#include<pybind11/stl.h>
#include<cmath>
#include "../../classes/graph.h"
struct mst_Edge{
    double wt;
    node_t start_node,end_node;
    edge_attr_dict_factory edge_attr;
};
py::object cpp_prim_mst_edges(py::object G, py::object minimum, py::object weight, py::object data, py::object ignore_nan) {
    Graph& G_=G.cast<Graph&>();
    node_dict_factory nodes_list = G_.node;
    std::unordered_set<node_t> nodes;
    
    for (node_dict_factory::iterator iter = nodes_list.begin(); iter != nodes_list.end(); iter++) {
        node_t node_id = iter->first;
        nodes.emplace(node_id);
    }
    int sign=1;
    if(minimum){
        std::cout<<"minimum:"<<minimum<<std::endl;
        sign=-1;
    }
    while(!nodes.empty()){
        // 随机从nodes中pop出一个元素
        node_t u=*nodes.begin();
        nodes.erase(nodes.begin());
        std::vector<mst_Edge> frontier;
        std::unordered_map<node_t,bool> visited;
        visited.emplace(std::make_pair(u,true));
        adj_attr_dict_factory u_neighbors=G_.adj[u];
        for(adj_attr_dict_factory::iterator i=u_neighbors.begin();i!=u_neighbors.end();i++){
            node_t v=i->first;
            edge_attr_dict_factory edge_attr=i->second;
            double wt;
            std::cout<<"weight:"<<py::cast<std::string>(weight)<<std::endl;
            if(edge_attr.find(py::cast<std::string>(weight))!=edge_attr.end()){
                wt=edge_attr[py::cast<std::string>(weight)]*sign;
            }
            else{
                wt=sign;
            }
            if(isnan(wt)){
                //
                if(ignore_nan){
                    continue;
                }
                PyErr_Format(PyExc_ValueError,"NaN found as an edge weight. Edge {(%R %R %R)}",G_.id_to_node.attr("get")(u),G_.id_to_node.attr("get")(v),);
            }
            
        }
    }


}