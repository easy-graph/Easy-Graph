import math
import random

import easygraph as eg
import numpy as np

from easygraph.utils import *


__all__ = [
    "sum_of_shortest_paths",
    "nodes_of_max_cc_without_shs",
    "structural_hole_influence_index",
]


@not_implemented_for("multigraph")
def sum_of_shortest_paths(G, S):
    r"""Returns the difference between the sum of lengths of all pairs shortest paths in G and the one in G\S.
    The experiment ml_metrics in [1]_

    Parameters
    ----------
    G: easygraph.Graph or easygraph.DiGraph

    S: list of int
        A list of nodes witch are structural hole spanners.

    Returns
    -------
    differ_between_sum : int
        The difference between the sum of lengths of all pairs shortest paths in G and the one in G\S.
        C(G/S)-C(G)

    Examples
    --------
    >>> G_t=eg.datasets.get_graph_blogcatalog()
    >>> S_t=eg.AP_Greedy(G_t, 10000)
    >>> diff = sum_of_shortest_paths(G_t, S_t)
    >>> print(diff)

    References
    ----------
    .. [1] https://dl.acm.org/profile/81484650642

    """
    mat_G = eg.Floyd(G)
    sum_G = 0
    inf_const_G = math.ceil((G.number_of_nodes() ** 3) / 3)
    for i in mat_G.values():
        for j in i.values():
            if math.isinf(j):
                j = inf_const_G
            sum_G += j
    G_S = G.copy()
    G_S.remove_nodes(S)
    mat_G_S = eg.Floyd(G_S)
    sum_G_S = 0
    inf_const_G_S = math.ceil((G_S.number_of_nodes() ** 3) / 3)
    for i in mat_G_S.values():
        for j in i.values():
            if math.isinf(j):
                j = inf_const_G_S
            sum_G_S += j
    return sum_G_S - sum_G


@not_implemented_for("multigraph")
def nodes_of_max_cc_without_shs(G, S):
    r"""Returns the number of nodes in the maximum connected component in graph G\S.
    The experiment ml_metrics in [1]_

    Parameters
    ----------
    G: easygraph.Graph or easygraph.DiGraph

    S: list of int
        A list of nodes witch are structural hole spanners.

    Returns
    -------
    G_S_nodes_of_max_CC: int
        The number of nodes in the maximum connected component in graph G\S.

    Examples
    --------
    >>> G_t=eg.datasets.get_graph_blogcatalog()
    >>> S_t=eg.AP_Greedy(G_t, 10000)
    >>> maxx = nodes_of_max_cc_without_shs(G_t, S_t)
    >>> print(maxx)

    References
    ----------
    .. [1] https://dl.acm.org/profile/81484650642

    """
    G_S = G.copy()
    G_S.remove_nodes(S)
    ccs = eg.connected_components(G_S)
    max_num = 0
    for cc in ccs:
        if len(cc) > max_num:
            max_num = len(cc)
    return max_num


class NodeParams:
    def __init__(self, active, inWeight, threshold):
        self.active = active
        self.inWeight = inWeight
        self.threshold = threshold


@not_implemented_for("multigraph")
def structural_hole_influence_index(
    G_original,
    S,
    C,
    model,
    variant=False,
    seedRatio=0.05,
    randSeedIter=10,
    countIterations=100,
    Directed=True,
):
    """Returns the SHII metric of each seed.

    Parameters
    ----------
    G_original: easygraph.Graph or easygraph.DiGraph

    S: list of int
        A list of nodes which are structural hole spanners.

    C: list of list
        Each list includes the nodes in one community.

    model: string
        Propagation Model. Should be IC or LT.

    variant: bool, default is False
        Whether returns variant SHII ml_metrics or not.
        variant SHII = # of the influenced outsider / # of the influenced insiders
        SHII = # of the influenced outsiders / # of the total influenced nodes

    seedRatio: float, default is 0.05
        # of sampled seeds / # of nodes of the community that the given SHS belongs to.

    randSeedIter: int, default is 10
        How many iterations to sample seeds.

    countIterations: int default is 100
        Number of monte carlo simulations to be used.

    Directed: bool, default is True
        Whether the graph is directed or not.

    Returns
    -------
    seed_shii_pair : dict
        the SHII metric of each seed

    Examples
    --------
    # >>> structural_hole_influence_index(G, [3, 20, 9], Com, 'LT', seedRatio=0.1, Directed=False)

    References
    ----------
    .. [1] https://dl.acm.org/doi/pdf/10.1145/2939672.2939807
    .. [2] https://github.com/LifangHe/KDD16_HAM/tree/master/SHII_metric

    """
    if not Directed:
        G = eg.DiGraph()
        for edge in G_original.edges:
            G.add_edge(edge[0], edge[1])
            G.add_edge(edge[1], edge[0])
    else:
        G = G_original.copy()
    # form pair like {node_1:community_label_1,node_2:community_label_2}
    node_label_pair = {}
    for community_label in range(len(C)):
        for node_i in range(len(C[community_label])):
            node_label_pair[C[community_label][node_i]] = community_label
    # print(node_label_pair)
    seed_shii_pair = {}
    for community_label in range(len(C)):
        nodesInCommunity = []
        seedSetInCommunity = []
        for node in node_label_pair.keys():
            if node_label_pair[node] == community_label:
                nodesInCommunity.append(node)
                if node in S:
                    seedSetInCommunity.append(node)

        seedSetSize = int(math.ceil(len(nodesInCommunity) * seedRatio))

        if len(seedSetInCommunity) == 0:
            continue

        for seed in seedSetInCommunity:
            print(">>>>>> processing seed ", seed, " now.")
            oneSeedSet = []
            if node not in oneSeedSet:
                oneSeedSet.append(seed)
            seedNeighborSet = []
            # using BFS to add neighbors of the SH spanner to the seedNeighborSet as seed candidates
            queue = []
            queue.append(seed)
            while len(queue) > 0:
                cur_node = queue[0]
                count_neighbor = 0
                for neighbor in G.neighbors(node=cur_node):
                    if neighbor not in seedNeighborSet:
                        seedNeighborSet.append(neighbor)
                    count_neighbor = count_neighbor + 1
                if count_neighbor > 0:
                    if (
                        len(queue) == 1
                        and len(oneSeedSet) + len(seedNeighborSet) < seedSetSize
                    ):
                        for node in seedNeighborSet:
                            if node not in oneSeedSet:
                                oneSeedSet.append(node)
                            queue.append(node)
                        seedNeighborSet.clear()
                queue.pop(0)

            avg_censor_score_1 = 0.0
            avg_censor_score_2 = 0.0

            for randIter in range(randSeedIter):
                if randIter % 5 == 0:
                    print("seed ", seed, ": ", randIter, " in ", randSeedIter)
                randSeedSet = []
                for node in oneSeedSet:
                    randSeedSet.append(node)
                seedNeighbors = []
                for node in seedNeighborSet:
                    seedNeighbors.append(node)
                while len(seedNeighbors) > 0 and len(randSeedSet) < seedSetSize:
                    r = random.randint(0, len(seedNeighbors) - 1)
                    if seedNeighbors[r] not in randSeedSet:
                        randSeedSet.append(seedNeighbors[r])
                    seedNeighbors.pop(r)

                if model == "IC":
                    censor_score_1, censor_score_2 = _independent_cascade(
                        G,
                        randSeedSet,
                        community_label,
                        countIterations,
                        node_label_pair,
                    )
                elif model == "LT":
                    censor_score_1, censor_score_2 = _linear_threshold(
                        G,
                        randSeedSet,
                        community_label,
                        countIterations,
                        node_label_pair,
                    )
                avg_censor_score_1 += censor_score_1 / randSeedIter
                avg_censor_score_2 += censor_score_2 / randSeedIter
                # print("seed ", seed, " avg_censor_score in ", randIter, "is ", censor_score_1 / randSeedIter)
            if variant:
                seed_shii_pair[seed] = avg_censor_score_2
            else:
                seed_shii_pair[seed] = avg_censor_score_1
    return seed_shii_pair


def _independent_cascade(G, S, community_label, countIterations, node_label_pair):
    avg_result_1 = 0
    avg_result_2 = 0
    N = G.number_of_nodes()
    for b in range(countIterations):
        # print(b, " in ", countIterations)
        p_vw = np.zeros((N, N))  # 节点被激活时，激活其它节点的概率,a对b的影响等于b对a的影响
        for random_i in range(N):
            for random_j in range(random_i + 1, N):
                num = random.random()
                p_vw[random_i][random_j] = num
                p_vw[random_j][random_i] = num
        Q = []
        activeNodes = []
        for v in S:
            Q.append(v)
            activeNodes.append(v)
        while len(Q) > 0:
            v = Q[0]
            for neighbor in G.neighbors(node=v):
                if neighbor not in activeNodes:
                    toss = random.random() + 0.1
                    if v <= 0 or neighbor <= 0:
                        print(v, neighbor)
                    # if toss>0.5:
                    #     activeNodes.append(neighbor)
                    #     Q.append(neighbor)
                    if toss >= p_vw[v - 1][neighbor - 1]:
                        activeNodes.append(neighbor)
                        Q.append(neighbor)
            Q.pop(0)
        self_cov = 0
        total_cov = 0
        uniqueActiveNodes = []
        for i in activeNodes:
            if i not in uniqueActiveNodes:
                uniqueActiveNodes.append(i)
        for v in uniqueActiveNodes:
            total_cov += 1
            if node_label_pair[v] == community_label:
                self_cov += 1
        censor_score_1 = (total_cov - self_cov) / total_cov
        censor_score_2 = (total_cov - self_cov) / self_cov
        avg_result_1 += censor_score_1 / countIterations
        avg_result_2 += censor_score_2 / countIterations
    return avg_result_1, avg_result_2


def _linear_threshold(G, S, community_label, countIterations, node_label_pair):
    tol = 0.00001
    avg_result_1 = 0
    avg_result_2 = 0
    for b in range(countIterations):
        activeNodes = []
        # T is the set of nodes that are to be processed
        T = []
        Q = {}
        for v in S:
            activeNodes.append(v)
            for neighbor in G.neighbors(node=v):
                if neighbor not in S:
                    weight_degree = 1.0 / float(G.in_degree()[neighbor])
                    if neighbor not in Q.keys():
                        np = NodeParams(False, weight_degree, random.random())
                        Q[neighbor] = np
                        T.append(neighbor)
                    else:
                        Q[neighbor].inWeight += weight_degree

        while len(T) > 0:
            u = T[0]
            if Q[u].inWeight >= Q[u].threshold + tol and not Q[u].active:
                activeNodes.append(u)
                Q[u].active = True
                for neighbor in G.neighbors(node=u):
                    if neighbor in S:
                        continue
                    weight_degree = 1.0 / float(G.in_degree()[neighbor])
                    if neighbor not in Q.keys():
                        np = NodeParams(False, weight_degree, random.random())
                        Q[neighbor] = np
                        T.append(neighbor)
                    else:
                        if not Q[neighbor].active:
                            T.append(neighbor)
                            Q[neighbor].inWeight += weight_degree
                            if Q[neighbor].inWeight - 1 > tol:
                                print("Error: the inweight for a node is > 1.")
            T.pop(0)

        T.clear()
        Q.clear()

        self_cov = 0
        total_cov = 0
        uniqueActiveNodes = []
        for i in activeNodes:
            if i not in uniqueActiveNodes:
                uniqueActiveNodes.append(i)
        for v in uniqueActiveNodes:
            total_cov += 1
            if node_label_pair[v] == community_label:
                self_cov += 1
        censor_score_1 = (total_cov - self_cov) / total_cov  # ==> SHII
        censor_score_2 = (total_cov - self_cov) / self_cov
        avg_result_1 += censor_score_1 / countIterations
        avg_result_2 += censor_score_2 / countIterations
    return avg_result_1, avg_result_2


if __name__ == "__main__":
    G = eg.datasets.get_graph_karateclub()
    Com = []
    t1 = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 17, 18, 20, 22]
    Com.append(t1)
    t2 = [9, 10, 15, 16, 19, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
    Com.append(t2)
    print("community_label:", Com)
    result = structural_hole_influence_index(
        G, [3, 20, 9], Com, "IC", seedRatio=0.1, Directed=False
    )
    print(result)
