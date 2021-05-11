from easygraph.functions.path import *

__all__ = [
    'closeness_centrality',
]

def closeness_centrality(G, weight=None):
    '''Compute closeness centrality for nodes.

    .. math::

        C_{WF}(u) = \frac{n-1}{N-1} \frac{n - 1}{\sum_{v=1}^{n-1} d(v, u)},
    
    Notice that the closeness distance function computes the 
    outcoming distance to `u` for directed graphs. To use 
    incoming distance, act on `G.reverse()`.

    Parameters
    ----------
    G : graph
      A easygraph graph

    weight : None or string, optional (default=None)
      If None, all edge weights are considered equal.
      Otherwise holds the name of the edge attribute used as weight.

    Returns
    -------
    nodes : dictionary
      Dictionary of nodes with closeness centrality as the value.

    '''
    result_dict = dict()
    nodes = G.nodes
    length = len(nodes)
    import functools
    if weight is not None:
        path_length = functools.partial(single_source_dijkstra, weight=weight)
    else:
        path_length = functools.partial(single_source_bfs)
    for node in nodes:
        x = path_length(G, node)
        dist = sum(x.values())
        cnt = len(x)
        if dist  == 0:
            result_dict[node] = 0
        else:
            result_dict[node] = (cnt-1)*(cnt-1)/(dist*(length-1))
    return result_dict
