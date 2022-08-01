#include "Path.h"
#include "Utils.h"
#include <queue>
#include <vector>


py::object _dijkstra_multisource(py::object G, py::object sources, py::object weight, py::object target) {
	Graph& G_ = py::extract<Graph&>(G);
	std::string weight_key = weight_to_string(weight);
	Graph::node_t target_id = py::extract<Graph::node_t>(G_.node_to_id.get(target, -1));
	std::map<Graph::node_t, Graph::weight_t> dist, seen;
	std::priority_queue<std::pair<Graph::weight_t, Graph::node_t>, std::vector<std::pair<Graph::weight_t, Graph::node_t>> > Q;
	py::list sources_list = py::list(sources);
	for (int i = 0;i < py::len(sources_list);i++) {
		Graph::node_t source = py::extract<Graph::node_t>(G_.node_to_id[sources_list[i]]);
		seen[source] = 0;
		Q.push(std::make_pair(0, source));
	}
	while (!Q.empty()) {
		std::pair<Graph::weight_t, Graph::node_t> node = Q.top();
		Q.pop();
		Graph::weight_t d = node.first;
		Graph::node_t v = node.second;
		if (dist.count(v)) {
			continue;
		}
		dist[v] = d;
		if (v == target_id) {
			break;
		}
		Graph::adj_dict_factory& adj = G_.adj;
		for (auto& neighbor_info : adj[v]) {
			Graph::node_t u = neighbor_info.first;
			Graph::weight_t cost = neighbor_info.second.count(weight_key) ? neighbor_info.second[weight_key] : 1;
			Graph::weight_t vu_dist = dist[v] + cost;
			if (dist.count(u)) {
				if (vu_dist < dist[u]) {
					PyErr_Format(PyExc_ValueError, "Contradictory paths found: negative weights?");
					return py::object();
				}
			}
			else if (!seen.count(u) || vu_dist < seen[u]) {
				seen[u] = vu_dist;
				Q.push(std::make_pair(vu_dist, u));
			}
			else {
				continue;
			}
		}
	}
	py::dict pydist = py::dict();
	for (const auto& kv : dist) {
		pydist[G_.id_to_node[kv.first]] = kv.second;
	}
	return pydist;
}

py::object Prim(py::object G){
	std::unordered_map<Graph::node_t,std::unordered_map<Graph::node_t,Graph::weight_t>> res_dict;
	py::dict result_dict=py::dict();
	Graph& G_=py::extract<Graph&>(G);
	Graph::adj_dict_factory adj=G_.adj;
	std::vector<Graph::node_t> selected;
	std::vector<Graph::node_t> candidate;
	Graph::node_dict_factory node_list=G_.node;
	for(Graph::node_dict_factory::iterator i=node_list.begin();i!=node_list.end();i++){
		Graph::node_t node_id=i->first;
		result_dict[G_.id_to_node[node_id]]=py::dict();
		if(selected.size()==0){
			selected.emplace_back(node_id);
		}
		else{
			candidate.emplace_back(node_id);
		}
	}
	while(candidate.size()>0){
		Graph::node_t start_id=-1;
		Graph::node_t end_id=-1;
		Graph::weight_t min_weight=INFINITY;
		for(Graph::node_t i=0;i<selected.size();i++){
			for(Graph::node_t j=0;j<candidate.size();j++){
				Graph::adj_attr_dict_factory node_adj=G_.adj[selected[i]];
				Graph::edge_attr_dict_factory edge_attr;
				Graph::weight_t edge_weight=INFINITY;
				bool j_exist=false;
				if(node_adj.find(candidate[j])!=node_adj.end()){
					edge_attr=node_adj[candidate[j]];
					edge_weight=edge_attr.find("weight")!=edge_attr.end()?edge_attr["weight"]:1;
					j_exist=true;
				}
				if((node_list.find(selected[i])!=node_list.end())&&
				j_exist&&
				(edge_weight<min_weight)){
					start_id = selected[i];
                    end_id = candidate[j];
                    min_weight = edge_weight;
				}
			}
		}
		if(start_id!=-1&&end_id!=-1){
			res_dict[start_id][end_id]=min_weight;
			selected.emplace_back(end_id);
			std::vector<Graph::node_t>::iterator temp_iter;
			temp_iter=std::find(candidate.begin(),candidate.end(),end_id);
			candidate.erase(temp_iter);
		}
		else{
			break;
		}
	}
	for(std::unordered_map<Graph::node_t,std::unordered_map<Graph::node_t,Graph::weight_t>>::iterator k=res_dict.begin();
	k!=res_dict.end();k++){
		py::object res_node=G_.id_to_node[k->first];
		for(std::unordered_map<Graph::node_t,Graph::weight_t>::iterator z=k->second.begin();z!=k->second.end();z++){
			py::object res_adj_node=G_.id_to_node[z->first];
			result_dict[res_node][res_adj_node]=z->second;
		}
	}
	return result_dict;

}
bool comp(const std::pair<std::pair<Graph::node_t,Graph::node_t>,Graph::weight_t> &a,const std::pair<std::pair<Graph::node_t,Graph::node_t>,Graph::weight_t> &b){
	return a.second < b.second;
}
py::object Kruskal(py::object G){
	std::unordered_map<Graph::node_t,std::unordered_map<Graph::node_t,Graph::weight_t>> res_dict;
	py::dict result_dict=py::dict();
	std::vector<std::vector<Graph::node_t>> group;
	Graph& G_=py::extract<Graph&>(G);
	Graph::adj_dict_factory adj=G_.adj;
	Graph::node_dict_factory node_list=G_.node;
	std::vector<std::pair<std::pair<Graph::node_t,Graph::node_t>,Graph::weight_t>>edge_list;
	for(Graph::node_dict_factory::iterator i=node_list.begin();i!=node_list.end();i++){
		Graph::node_t i_id=i->first;
		result_dict[G_.id_to_node[i_id]]=py::dict();
		std::vector<Graph::node_t> temp_vector;
		temp_vector.emplace_back(i_id);
		group.emplace_back(temp_vector);
		Graph::adj_attr_dict_factory i_adj=adj[i_id];
		for(Graph::adj_attr_dict_factory::iterator j=i_adj.begin();j!=i_adj.end();j++){
			Graph::node_t j_id=j->first;
			Graph::weight_t weight=adj[i_id][j_id].find("weight")!=adj[i_id][j_id].end()?adj[i_id][j_id]["weight"]:1;
			edge_list.emplace_back(std::make_pair(std::make_pair(i_id,j_id),weight));
		}
	}
	std::sort(edge_list.begin(),edge_list.end(),comp);
	Graph::node_t m,n;
	for(auto edge:edge_list){
		for(Graph::node_t i=0;i<group.size();i++){
			for(Graph::node_t j=0;j<group[i].size();j++){
				if(group[i][j]==edge.first.first){
					m=i;
					break;
				}
			}
			for(Graph::node_t j=0;j<group[i].size();j++){
				if(group[i][j]==edge.first.second){
					n=i;
					break;
				}
			}
		}
		if(m!=n){

			res_dict[edge.first.first][edge.first.second]=edge.second;
			std::vector<Graph::node_t> temp_vector;
			group[m].insert(group[m].end(),group[n].begin(),group[n].end());
			group[n].clear();
		}
	}
	for(std::unordered_map<Graph::node_t,std::unordered_map<Graph::node_t,Graph::weight_t>>::iterator k=res_dict.begin();
	k!=res_dict.end();k++){
		py::object res_node=G_.id_to_node[k->first];
		for(std::unordered_map<Graph::node_t,Graph::weight_t>::iterator z=k->second.begin();z!=k->second.end();z++){
			py::object res_adj_node=G_.id_to_node[z->first];
			result_dict[res_node][res_adj_node]=z->second;
		}
	}
	return result_dict;
}

py::object Floyd(py::object G){
	std::unordered_map<Graph::node_t,std::unordered_map<Graph::node_t,Graph::weight_t>> res_dict;
	Graph& G_=py::extract<Graph&>(G);
	Graph::adj_dict_factory adj=G_.adj;
	py::dict result_dict=py::dict();
	Graph::node_dict_factory node_list=G_.node;
	for(Graph::node_dict_factory::iterator i=node_list.begin();i!=node_list.end();i++){
		result_dict[G_.id_to_node[i->first]]=py::dict();
		Graph::adj_attr_dict_factory temp_key=adj[i->first];
		for(Graph::node_dict_factory::iterator j=node_list.begin();j!=node_list.end();j++){
			if(temp_key.find(j->first)!=temp_key.end()){
				if(adj[i->first][j->first].count("weight")==0){
					adj[i->first][j->first]["weight"]=1;
				}
				res_dict[i->first][j->first]=adj[i->first][j->first]["weight"];
			}
			else{
				res_dict[i->first][j->first]=INFINITY;
			}
			if(i->first==j->first){
				res_dict[i->first][i->first]=0;
			}
		}
	}

	for(Graph::node_dict_factory::iterator k=node_list.begin();k!=node_list.end();k++){
		for(Graph::node_dict_factory::iterator i=node_list.begin();i!=node_list.end();i++){
			for(Graph::node_dict_factory::iterator j=node_list.begin();j!=node_list.end();j++){
				Graph::weight_t temp=res_dict[i->first][k->first]+res_dict[k->first][j->first];
				Graph::weight_t i_j_weight=res_dict[i->first][j->first];
				if(i_j_weight>temp){
					res_dict[i->first][j->first]=temp;
				}
			}	
		}
	}
	
	for(std::unordered_map<Graph::node_t,std::unordered_map<Graph::node_t,Graph::weight_t>>::iterator k=res_dict.begin();
	k!=res_dict.end();k++){
		py::object res_node=G_.id_to_node[k->first];
		for(std::unordered_map<Graph::node_t,Graph::weight_t>::iterator z=k->second.begin();z!=k->second.end();z++){
			py::object res_adj_node=G_.id_to_node[z->first];
			result_dict[res_node][res_adj_node]=z->second;
		}
	}
	return result_dict;
}
