import math
import random
import sys

import easygraph as eg

from easygraph.functions.components.strongly_connected import condensation
from easygraph.functions.components.strongly_connected import (
    number_strongly_connected_components,
)
from easygraph.utils import *


__all__ = [
    "maxBlock",
    "maxBlockFast",
]
tim = 0
sys.setrecursionlimit(9000000)


class dom_g:
    def __init__(self, N, M):
        self.tot = 0
        self.h = []
        self.ne = []
        self.to = []
        for i in range(N + 1):
            self.h.append(0)
        for i in range(max(N + 1, M + 1)):
            self.ne.append(0)
            self.to.append(0)

    def add(self, x, y):
        self.tot += 1
        self.to[self.tot] = y
        self.ne[self.tot] = self.h[x]
        self.h[x] = self.tot


def _tarjan(x, dfn, repos, g, fa):
    global tim
    tim += 1
    dfn[x] = tim
    repos[tim] = x
    i = g.h[x]
    while i:
        if dfn[g.to[i]] == 0:
            fa[g.to[i]] = x
            _tarjan(g.to[i], dfn, repos, g, fa)
        i = g.ne[i]


def _find(x, f, dfn, semi, mi):
    if x == f[x]:
        return x
    tmp = f[x]
    f[x] = _find(f[x], f, dfn, semi, mi)
    if dfn[semi[mi[tmp]]] < dfn[semi[mi[x]]]:
        mi[x] = mi[tmp]
    return f[x]


def _dfs(x, tr, ans, desc_set):
    ans[x] += 1
    i = tr.h[x]

    while i:
        y = tr.to[i]
        desc_set[x].add(y)
        _dfs(y, tr, ans, desc_set)
        ans[x] += ans[y]
        for n in desc_set[y]:
            desc_set[x].add(n)
        i = tr.ne[i]


def _get_idom(G, G_tr, node_s, ans_real, desc_set_real):
    """Find the immediate dominator of each node and construct an s-rooted dominator tree.

    Parameters
    ----------
    G: easygraph.DiGraph

    G_tr: easygraph.DiGraph
        an s-rooted dominator tree to be constructed.

    node_s: int
        the node s

    ans_real: dict
        denotes the number of proper descendants nu of each node u in the dominator tree.
        a result to be calculated

    desc_set_real: dict
        denotes the set of proper descendants of node u in the dominator tree.
        a result to be calculated

    Examples
    --------
    # >>> G_tr = eg.DiGraph()
    # >>> n_set = {}
    # >>> desc_set = {}
    # >>> _get_idom(G, G_tr, node_s, n_set, desc_set)

    References
    ----------
    .. [1] http://keyblog.cn/article-173.html

    """

    global tim
    tim = 0
    n_dom = G.number_of_nodes()
    m_dom = G.number_of_edges()
    g = dom_g(n_dom + 1, m_dom + 1)
    rg = dom_g(n_dom + 1, m_dom + 1)
    ng = dom_g(n_dom + 1, m_dom + 1)
    tr = dom_g(n_dom + 1, m_dom + 1)

    dfn = [0 for i in range(n_dom + 1)]
    repos = [0 for i in range(n_dom + 1)]
    mi = [i for i in range(n_dom + 1)]
    fa = [0 for i in range(n_dom + 1)]
    f = [i for i in range(n_dom + 1)]
    semi = [i for i in range(n_dom + 1)]
    idom = [0 for i in range(n_dom + 1)]

    # init
    j = 0
    node_map = {}
    index_map = {}
    for node in G.nodes:
        j += 1
        node_map[node] = j
        index_map[j] = node

    for edge in G.edges:
        g.add(node_map[edge[0]], node_map[edge[1]])
        rg.add(node_map[edge[1]], node_map[edge[0]])

    # tarjan
    _tarjan(node_map[node_s], dfn, repos, g, fa)
    # work
    i = n_dom
    while i >= 2:
        x = repos[i]
        tmp = n_dom
        j = rg.h[x]
        while j:
            if dfn[rg.to[j]] == 0:
                j = rg.ne[j]
                continue
            if dfn[rg.to[j]] < dfn[x]:
                tmp = min(tmp, dfn[rg.to[j]])
            else:
                _find(rg.to[j], f, dfn, semi, mi)
                tmp = min(tmp, dfn[semi[mi[rg.to[j]]]])
            j = rg.ne[j]

        semi[x] = repos[tmp]
        f[x] = fa[x]
        ng.add(semi[x], x)
        x = repos[i - 1]
        j = ng.h[x]
        while j:
            y = ng.to[j]
            _find(y, f, dfn, semi, mi)
            if semi[mi[y]] == semi[y]:
                idom[y] = semi[y]
            else:
                idom[y] = mi[y]
            j = ng.ne[j]
        i -= 1

    i = 2
    while i <= n_dom:
        x = repos[i]
        if x != 0:
            if idom[x] != semi[x]:
                idom[x] = idom[idom[x]]
            tr.add(idom[x], x)
            if x != node_map[node_s]:
                G_tr.add_edge(index_map[idom[x]], index_map[x])
        i += 1
    G_tr.add_node(node_s)
    ans = {}
    desc_set = {}
    for node in G_tr.nodes:
        ans[node_map[node]] = 0
        desc_set[node_map[node]] = set()

    _dfs(node_map[node_s], tr, ans, desc_set)
    for key in ans.keys():
        ans[key] -= 1
        ans_real[index_map[key]] = ans[key]
    for key in desc_set.keys():
        desc_set_real[index_map[key]] = set()
        for value in desc_set[key]:
            desc_set_real[index_map[key]].add(index_map[value])


def _find_topk_shs_under_l(G, f_set, k, L):
    """Find the top-k structural hole spanners under L simulations.

    Parameters
    ----------
    G: easygraph.DiGraph

    f_set: dict
        user vi shares his/her information on network G at a rate fi.

    k: int
        top - k structural hole spanners.

    L: int
        the number of simulations.

    Returns
    -------
    S_list : list
        A set S of k nodes that block the maximum number of information propagations within L simulations.

    ave_H_Lt_S: float
        the average number of blocked information propagations by the nodes in set S with L t simulations.

    """
    h_set = {}
    n = G.number_of_nodes()
    for node in G.nodes:
        h_set[node] = 0
    for l in range(L):
        if l % 100000 == 0:
            print("[", l, "/", L, "] find topk shs under L")
        # Choose a node s from the n nodes in G randomly
        node_s = random.choice(list(G.nodes))
        # Generate a graph G & = (V, E & ) from G under the live-edge graph model
        G_live = G.copy()
        for edge in G_live.edges:
            wij = G_live[edge[0]][edge[1]]["weight"]
            toss = random.random() + 0.1
            if toss >= wij:
                G_live.remove_edge(edge[0], edge[1])
        # Obtain the induced subgraph by the set R G & (s ) of reachable nodes from s
        R_set = eg.connected_component_of_node(G_live, node_s)
        G_subgraph = eg.DiGraph()
        for node in R_set:
            G_subgraph.add_node(node)
        for edge in G_live.edges:
            if edge[0] in G_subgraph.nodes and edge[1] in G_subgraph.nodes:
                G_subgraph.add_edge(edge[0], edge[1])
        # Find the immediate dominator idom (v ) of each node v $ V && \ { s } in G
        # Construct an s -rooted dominator tree
        # Calculate the number of proper descendants n u of each node u $ V &&
        G_tr = eg.DiGraph()
        n_set = {}
        desc_set = {}
        _get_idom(G_subgraph, G_tr, node_s, n_set, desc_set)
        for node_u in G_tr.nodes:
            if node_u != node_s:
                # the number of blocked information propagations by node u
                h_set[node_u] += n_set[node_u] * f_set[node_s]
    ave_H_set = {}
    for node in G.nodes:
        ave_H_set[node] = h_set[node] * n / L
    ordered_set = sorted(ave_H_set.items(), key=lambda x: x[1], reverse=True)
    S_list = []
    ave_H_Lt_S = 0
    for i in range(k):
        S_list.append((ordered_set[i])[0])
        ave_H_Lt_S += (ordered_set[i])[1]
    return S_list, ave_H_Lt_S


def _get_estimated_opt(G, f_set, k, c, delta):
    """Estimation of the optimal value OPT.

    Parameters
    ----------
    G: easygraph.DiGraph

    f_set: dict
        user vi shares his/her information on network G at a rate fi.

    k: int
        top - k structural hole spanners.

    c: int
        Success probability 1-n^-c of maxBlock.

    delta: float
        a small value delta > 0.

    Returns
    -------
    res_opt : float
        An approximate value OPT.

    """
    print("Estimating the optimal value OPT...")
    n = G.number_of_nodes()
    opt_ub = 0
    for f_key in f_set.keys():
        opt_ub = opt_ub + f_set[f_key]
    opt_ub = opt_ub * k * (n - 1)
    T = math.log((opt_ub / (delta / 2)), 2)
    T = math.ceil(T)
    lamda = 4 * (c * math.log(n, 2) + math.log(k * T, 2)) * (2 * k + 1) * k * n * n
    for t in range(T):
        opt_g = opt_ub / math.pow(2, t + 1)
        L_t = math.ceil(lamda / opt_g)
        print("[", t, "/", T, "] Estimating OPT: L=", L_t)
        S_list, ave_H_Lt_S = _find_topk_shs_under_l(G, f_set, k, L_t)
        if ave_H_Lt_S >= opt_g:
            res_opt = opt_g / 2
            return res_opt
    print("[Warning] OPT is not greater that delta")
    return -1


def _find_separation_nodes(G):
    G_s = condensation(G)
    SCC_mapping = {}
    incoming_info = G_s.graph["incoming_info"]
    G_s_undirected = eg.Graph()
    sep_nodes = set()
    for node in (G_s.nodes).keys():
        SCC_mapping[node] = G_s.nodes[node]["member"]
        if len(G_s.nodes[node]["member"]) == 1:
            sep_nodes.add(node)
        G_s_undirected.add_node(node, member=G_s.nodes[node]["member"])
    for edge in G_s.edges:
        G_s_undirected.add_edge(edge[0], edge[1])
    cut_nodes = eg.generator_articulation_points(G_s_undirected)
    out_degree = G_s.out_degree()
    in_degree = G_s.in_degree()
    separations = set()
    for cut_node in cut_nodes:
        if cut_node in sep_nodes:
            if out_degree[cut_node] >= 1 and in_degree[cut_node] >= 1:
                CC_u = eg.connected_component_of_node(G_s_undirected, node=cut_node)
                G_CC = G_s_undirected.nodes_subgraph(list(CC_u))
                G_CC.remove_node(cut_node)
                successors = G_s.neighbors(node=cut_node)
                predecessors = G_s.predecessors(node=cut_node)
                CC_removal = eg.connected_components(G_CC)
                flag = True
                for group in CC_removal:
                    flag_succ = False
                    flag_pred = False
                    for node in group:
                        if node in successors:
                            flag_succ = True
                            if flag_pred:
                                flag = False
                                break
                        elif node in predecessors:
                            flag_pred = True
                            if flag_succ:
                                flag = False
                                break
                    if not flag:
                        break
                if flag:
                    separations.add(list(SCC_mapping[cut_node])[0])
    return separations, SCC_mapping, incoming_info


def _find_ancestors_of_node(G, node_t):
    G_reverse = eg.DiGraph()
    for node in G.nodes:
        G_reverse.add_node(node)
    for edge in G.edges:
        G_reverse.add_edge(edge[1], edge[0])
    node_dict = eg.Dijkstra(G_reverse, node=node_t)
    ancestors = []
    for node in G.nodes:
        if node_dict[node] < float("inf") and node != node_t:
            ancestors.append(node)
    return ancestors


@not_implemented_for("multigraph")
def maxBlock(G, k, f_set=None, delta=1, eps=0.5, c=1, flag_weight=False):
    """Structural hole spanners detection via maxBlock method.

    Parameters
    ----------
    G: easygraph.DiGraph

    k: int
        top - k structural hole spanners.

    f_set: dict, optional
        user vi shares his/her information on network G at a rate fi.
        default is a random [0,1) integer for each node

    delta: float, optional (default: 1)
        a small value delta > 0.

    eps: float, optional (default: 0.5)
        an error ratio eps with 0 < eps < 1.

    c: int, optional (default: 1)
        Success probability 1-n^-c of maxBlock.

    flag_weight: bool, optional (default: False)
        Denotes whether each edge has attribute 'weight'

    Returns
    -------
    S_list : list
        The list of each top-k structural hole spanners.

    See Also
    -------
    maxBlockFast

    Examples
    --------
    # >>> maxBlock(G, 100)

    References
    ----------
    .. [1] https://doi.org/10.1016/j.ins.2019.07.072

    """
    if f_set is None:
        f_set = {}
        for node in G.nodes:
            f_set[node] = random.random()
    if not flag_weight:
        for edge in G.edges:
            G[edge[0]][edge[1]]["weight"] = random.random()
    n = G.number_of_nodes()
    approximate_opt = _get_estimated_opt(G, f_set, k, c, delta)
    print("approximate_opt:", approximate_opt)
    L_min = (k + c) * math.log(n, 2) + math.log(4, 2)
    L_min = L_min * k * n * n * math.pow(eps, -2) * (8 * k + 2 * eps)
    L_min = L_min / approximate_opt
    L_min = math.ceil(L_min)
    print("L_min:", L_min)
    S_list, ave_H_Lt_S = _find_topk_shs_under_l(G, f_set, k, L_min)
    return S_list


@not_implemented_for("multigraph")
def maxBlockFast(G, k, f_set=None, L=None, flag_weight=False):
    """Structural hole spanners detection via maxBlockFast method.

    Parameters
    ----------
    G: easygraph.DiGraph

    G: easygraph.DiGraph

    k: int
        top - k structural hole spanners.

    f_set: dict, optional
        user vi shares his/her information on network G at a rate fi.
        default is a random [0,1) integer for each node

    L: int, optional (default: log2n)
        Simulation time L for maxBlockFast.

    flag_weight: bool, optional (default: False)
        Denotes whether each edge has attribute 'weight'

    See Also
    -------
    maxBlock

    Examples
    --------
    # >>> maxBlockFast(G, 100)

    References
    ----------
    .. [1] https://doi.org/10.1016/j.ins.2019.07.072

    """
    h_set = {}
    n = G.number_of_nodes()
    if L is None:
        L = math.ceil(math.log(n, 2))
    # print("L:", L)
    if f_set is None:
        f_set = {}
        for node in G.nodes:
            f_set[node] = random.random()
    for node in G.nodes:
        h_set[node] = 0
    if not flag_weight:
        for edge in G.edges:
            G[edge[0]][edge[1]]["weight"] = random.random()
    for l in range(L):
        if l % 10000 == 0:
            print(l, "/", L, "...")
        # Generate a graph G & = (V, E & ) from G under the live-edge graph model
        G_live = G.copy()
        for edge in G_live.edges:
            wij = G_live[edge[0]][edge[1]]["weight"]
            toss = random.random() + 0.1
            if toss >= wij:
                G_live.remove_edge(edge[0], edge[1])

        G0 = G_live.copy()
        d_dict = {}
        ns = number_strongly_connected_components(G0)
        non_considered_nodes = set()
        for node in G0.nodes:
            d_dict[node] = 1
            non_considered_nodes.add(node)
        G_p_1 = G0.copy()
        for i in range(ns):
            separation_nodes, SCC_mapping, incoming_info = _find_separation_nodes(G_p_1)
            # print("separation_nodes:", separation_nodes)
            if len(separation_nodes) > 0:
                chosen_node = -1
                for node in separation_nodes:
                    node_dict = eg.Dijkstra(G_p_1, node=node)
                    flag = True
                    for other_sep in separation_nodes:
                        if other_sep != node:
                            if node_dict[other_sep] < float("inf"):
                                flag = False
                                break
                    if flag:
                        chosen_node = node
                        break
                # print("chosen_node:", chosen_node)
                G_tr = eg.DiGraph()
                n_set = {}
                desc_set = {}
                _get_idom(G_p_1, G_tr, chosen_node, n_set, desc_set)
                ancestors = _find_ancestors_of_node(G_p_1, chosen_node)
                sum_fi = 0
                for node_av in ancestors:
                    sum_fi += f_set[node_av]
                for node_u in G_tr.nodes:
                    D_u = 0
                    for desc in desc_set[node_u]:
                        if desc not in d_dict.keys():
                            print(
                                "Error: desc:",
                                desc,
                                "node_u",
                                node_u,
                                "d_dict:",
                                d_dict,
                            )
                            print(desc_set[node_u])
                        D_u += d_dict[desc]
                    if node_u != chosen_node:
                        h_set[node_u] += (f_set[chosen_node] + sum_fi) * D_u
                    elif node_u == chosen_node:
                        h_set[node_u] += sum_fi * D_u
                d_dict[chosen_node] = 0
                for node_vj in G_tr.nodes:
                    d_dict[chosen_node] += d_dict[node_vj]
                G_p = G_p_1.copy()
                for neighbor in G_p_1.neighbors(node=chosen_node):
                    G_p.remove_edge(chosen_node, neighbor)
                G_p_1 = G_p.copy()
                non_considered_nodes.remove(chosen_node)
            else:
                V_set = set()
                for key in SCC_mapping.keys():
                    for node in SCC_mapping[key]:
                        if (node in non_considered_nodes) and (
                            node not in incoming_info.keys()
                        ):
                            V_set.add(node)
                    if len(V_set) > 0:
                        break
                # print("V_set:", V_set)
                for node_v in V_set:
                    G_tr = eg.DiGraph()
                    n_set = {}
                    desc_set = {}
                    _get_idom(G_p_1, G_tr, node_v, n_set, desc_set)
                    for node_u in G_tr.nodes:
                        D_u = 0
                        for desc in desc_set[node_u]:
                            if desc not in d_dict.keys():
                                print(
                                    "Error: desc:",
                                    desc,
                                    "node_u",
                                    node_u,
                                    "d_dict:",
                                    d_dict,
                                )
                                print(desc_set[node_u])
                            D_u += d_dict[desc]
                        h_set[node_u] += f_set[node_v] * D_u
                G_p = G_p_1.copy()
                for node_v in V_set:
                    non_considered_nodes.remove(node_v)
                    for neighbor in G_p_1.neighbors(node=node_v):
                        G_p.remove_edge(node_v, neighbor)
                G_p_1 = G_p.copy()
    ave_H_set = {}
    for node in G.nodes:
        ave_H_set[node] = h_set[node] * n / L
    ordered_set = sorted(ave_H_set.items(), key=lambda x: x[1], reverse=True)
    S_list = []
    for i in range(k):
        S_list.append((ordered_set[i])[0])
    return S_list
