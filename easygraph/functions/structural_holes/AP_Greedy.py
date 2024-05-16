import math
import random

import easygraph as eg

from easygraph.functions.components.biconnected import generator_articulation_points
from easygraph.functions.components.connected import connected_components
from easygraph.utils.decorators import *


__all__ = ["common_greedy", "AP_Greedy"]


@not_implemented_for("multigraph")
@only_implemented_for_UnDirected_graph
def common_greedy(G, k, c=1.0, weight="weight"):
    """Common greedy method for structural hole spanners detection.

    Returns top k nodes as structural hole spanners,
    Algorithm 1 of [1]_

    Parameters
    ----------
    G : easygraph.Graph
        An undirected graph.

    k : int
        top - k structural hole spanners

    c : float, optional (default : 1.0)
        To define zeta: zeta = c * (n*n*n), and zeta is the large
        value assigned as the shortest distance of two unreachable
        vertices.
        Default is 1.

    weight : String or None, optional (default : 'weight')
        Key for edge weight. None if not concerning about edge weight.

    Returns
    -------
    common_greedy : list
        The list of each top-k structural hole spanners.

    See Also
    --------
    AP_Greedy

    Examples
    --------
    Returns the top k nodes as structural hole spanners, using **common_greedy**.

    >>> common_greedy(G,
    ...               k = 3, # To find top three structural holes spanners.
    ...               c = 1.0, # To define zeta: zeta = c * (n*n*n), and zeta is the large value assigned as the shortest distance of two unreachable vertices.
    ...               weight = 'weight')

    References
    ----------
    .. [1] https://dl.acm.org/profile/81484650642

    """
    v_sns = []
    G_i = G.copy()
    N = len(G)
    for i in range(k):
        sorted_nodes = sort_nodes_by_degree(G_i, weight)
        C_max = 0

        for j in range(N - i):
            G_i_j = G_i.copy()
            G_i_j.remove_node(sorted_nodes[j])
            upper_bound = procedure1(G_i_j, c)
            if upper_bound < C_max:
                pass
            else:
                sum_all_shortest_paths = procedure2(G_i_j, c)
                if sum_all_shortest_paths >= C_max:
                    v_i = sorted_nodes[j]
                    C_max = sum_all_shortest_paths
                else:
                    pass
            del G_i_j

        v_sns.append(v_i)
        G_i.remove_node(v_i)

    del G_i
    return v_sns


def sort_nodes_by_degree(G, weight="weight"):
    sorted_nodes = []
    for node, degree in sorted(
        G.degree(weight=weight).items(), key=lambda x: x[1], reverse=True
    ):
        sorted_nodes.append(node)
    return sorted_nodes


def procedure1(G, c=1.0):
    """
    Procedure 1 of https://dl.acm.org/profile/81484650642

    Parameters
    -----------
    G : graph

    c : float
        To define zeta: zeta = c * (n*n*n)
        Default is 1.

    """
    components = connected_components(G)
    upper_bound = 0
    for component in components:
        component_subgraph = G.nodes_subgraph(from_nodes=list(component))
        spanning_tree = _get_spanning_tree_of_component(component_subgraph)

        random_root = list(spanning_tree.nodes)[
            random.randint(0, len(spanning_tree) - 1)
        ]
        num_subtree_nodes = _get_num_subtree_nodes(spanning_tree, random_root)

        N_tree = num_subtree_nodes[random_root]
        for node, num in num_subtree_nodes.items():
            upper_bound += 2 * num * (N_tree - num)

        del component_subgraph, spanning_tree

    N_G = len(G)
    zeta = c * math.pow(N_G, 3)
    for component in components:
        N_c = len(component)
        upper_bound += N_c * (N_G - N_c) * zeta

    return upper_bound


def _get_spanning_tree_of_component(G):
    spanning_tree = eg.Graph()
    seen = set()

    def _plain_dfs(u):
        for v, edge_data in G.adj[u].items():
            if v not in seen:
                seen.add(v)
                spanning_tree.add_edge(u, v)
                _plain_dfs(v)

    random_node = list(G.nodes)[0]
    seen.add(random_node)
    spanning_tree.add_node(random_node)

    _plain_dfs(random_node)

    return spanning_tree


def _get_num_subtree_nodes(G, root):
    num_subtree_nodes = dict()
    seen = set()

    def _plain_dfs(u):
        num_nodes = 1
        for v, edge_data in G.adj[u].items():
            if v not in seen:
                seen.add(v)
                num_nodes += _plain_dfs(v)

        num_subtree_nodes[u] = num_nodes
        return num_nodes

    seen.add(root)
    _plain_dfs(root)

    return num_subtree_nodes


def procedure2(G, c=1.0):
    """
    Procedure 2 of https://dl.acm.org/profile/81484650642

    Parameters
    -----------
    G : graph

    c : float
        To define zeta: zeta = c * (n*n*n)
        Default is 1.
    """
    components = connected_components(G)
    C = 0
    N_G = len(G)
    zeta = c * math.pow(N_G, 3)
    for component in components:
        component_subgraph = G.nodes_subgraph(from_nodes=list(component))
        C_l = _get_sum_all_shortest_paths_of_component(component_subgraph)
        N_c = len(component)
        C += C_l + N_c * (N_G - N_c) * zeta

        del component_subgraph

    return C


def _get_sum_all_shortest_paths_of_component(G):
    # TODO: Using randomized algorithm in http://de.arxiv.org/pdf/1503.08528
    #       instead of bfs method.
    def _plain_bfs(G, source):
        seen = {source}
        nextlevel = {source}
        level = 1
        sum_paths_of_G = 0

        while nextlevel:
            thislevel = nextlevel
            nextlevel = set()
            for u in thislevel:
                for v in G.adj[u]:
                    if v not in seen:
                        seen.add(v)
                        nextlevel.add(v)
                        sum_paths_of_G += level
            level += 1
        return sum_paths_of_G

    sum_paths = 0
    for node in G.nodes:
        sum_paths += _plain_bfs(G, node)

    return sum_paths


@not_implemented_for("multigraph")
@only_implemented_for_UnDirected_graph
def AP_Greedy(G, k, c=1.0, weight="weight"):
    """AP greedy method for structural hole spanners detection.

    Returns top k nodes as structural hole spanners,
    Algorithm 2 of [1]_

    Parameters
    ----------
    G : easygraph.Graph
        An undirected graph.

    k : int
        top - k structural hole spanners

    c : float, optional (default : 1.0)
        To define zeta: zeta = c * (n*n*n), and zeta is the large
        value assigned as the shortest distance of two unreachable
        vertices.
        Default is 1.

    weight : String or None, optional (default : 'weight')
        Key for edge weight. None if not concerning about edge weight.

    Returns
    -------
    AP_greedy : list
        The list of each top-k structural hole spanners.

    Examples
    --------
    Returns the top k nodes as structural hole spanners, using **AP_greedy**.

    >>> AP_greedy(G,
    ...           k = 3, # To find top three structural holes spanners.
    ...           c = 1.0, # To define zeta: zeta = c * (n*n*n), and zeta is the large value assigned as the shortest distance of two unreachable vertices.
    ...           weight = 'weight')

    References
    ----------
    .. [1] https://dl.acm.org/profile/81484650642
    """
    v_sns = []
    G_i = G.copy()
    N = len(G)
    for i in range(k):
        v_ap, lower_bound = _get_lower_bound_of_ap_nodes(G_i, c)
        upper_bound = _get_upper_bound_of_non_ap_nodes(G_i, v_ap, c)
        lower_bound = sorted(lower_bound.items(), key=lambda x: x[1], reverse=True)

        # print(upper_bound)
        # print(lower_bound)
        if len(lower_bound) != 0 and lower_bound[0][1] > max(upper_bound):
            v_i = lower_bound[0][0]
        else:  # If acticulation points not chosen, use common_greedy instead.
            sorted_nodes = sort_nodes_by_degree(G_i, weight)
            C_max = 0

            for j in range(N - i):
                G_i_j = G_i.copy()
                G_i_j.remove_node(sorted_nodes[j])
                upper_bound = procedure1(G_i_j, c)
                if upper_bound < C_max:
                    pass
                else:
                    sum_all_shortest_paths = procedure2(G_i_j, c)
                    if sum_all_shortest_paths >= C_max:
                        v_i = sorted_nodes[j]
                        C_max = sum_all_shortest_paths
                    else:
                        pass
                del G_i_j

        v_sns.append(v_i)
        G_i.remove_node(v_i)

    del G_i
    return v_sns


def _get_lower_bound_of_ap_nodes(G, c=1.0):
    """
    Returns the articulation points and lower bound for each of them.
    Procedure 3 of https://dl.acm.org/profile/81484650642

    Parameters
    ----------
    G : graph
        An undirected graph.

    c : float
        To define zeta: zeta = c * (n*n*n), and zeta is the large
        value assigned as the shortest distance of two unreachable
        vertices.
        Default is 1.
    """
    v_ap = []
    lower_bound = dict()

    N_G = len(G)
    zeta = c * math.pow(N_G, 3)
    components = connected_components(G)
    for component in components:
        component_subgraph = G.nodes_subgraph(from_nodes=list(component))
        articulation_points = list(generator_articulation_points(component_subgraph))
        N_component = len(component_subgraph)
        for articulation in articulation_points:
            component_subgraph_after_remove = component_subgraph.copy()
            component_subgraph_after_remove.remove_node(articulation)

            lower_bound_value = 0
            lower_bound_value += sum(
                (len(temp) * (N_G - len(temp))) for temp in components
            )
            lower_bound_value += sum(
                (len(temp) * (N_component - 1 - len(temp)))
                for temp in connected_components(component_subgraph_after_remove)
            )
            lower_bound_value += 2 * N_component - 2 * N_G
            lower_bound_value *= zeta

            v_ap.append(articulation)
            lower_bound[articulation] = lower_bound_value

            del component_subgraph_after_remove

        del component_subgraph

    return v_ap, lower_bound


def _get_upper_bound_of_non_ap_nodes(G, ap: list, c=1.0):
    """
    Returns the upper bound value for each non-articulation points.
    Eq.(14) of https://dl.acm.org/profile/81484650642

    Parameters
    ----------
    G : graph
        An undirected graph.

    ap : list
        Articulation points of G.

    c : float
        To define zeta: zeta = c * (n*n*n), and zeta is the large
        value assigned as the shortest distance of two unreachable
        vertices.
        Default is 1.
    """
    upper_bound = []

    N_G = len(G)
    zeta = c * math.pow(N_G, 3)
    components = connected_components(G)
    for component in components:
        non_articulation_points = component - set(ap)
        for node in non_articulation_points:
            upper_bound_value = 0
            upper_bound_value += sum(
                (len(temp) * (N_G - len(temp))) for temp in components
            )
            upper_bound_value += 2 * len(component) + 1 - 2 * N_G
            upper_bound_value *= zeta

            upper_bound.append(upper_bound_value)

    return upper_bound
