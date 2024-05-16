import easygraph as eg


__all__ = [
    "predecessor",
]


def predecessor(G, source, target=None, cutoff=None, return_seen=None):
    """Returns dict of predecessors for the path from source to all nodes in G.

    Parameters
    ----------
    G : EasyGraph graph

    source : node label
       Starting node for path

    target : node label, optional
       Ending node for path. If provided only predecessors between
       source and target are returned

    cutoff : integer, optional
        Depth to stop the search. Only paths of length <= cutoff are returned.

    return_seen : bool, optional (default=None)
        Whether to return a dictionary, keyed by node, of the level (number of
        hops) to reach the node (as seen during breadth-first-search).

    Returns
    -------
    pred : dictionary
        Dictionary, keyed by node, of predecessors in the shortest path.


    (pred, seen): tuple of dictionaries
        If `return_seen` argument is set to `True`, then a tuple of dictionaries
        is returned. The first element is the dictionary, keyed by node, of
        predecessors in the shortest path. The second element is the dictionary,
        keyed by node, of the level (number of hops) to reach the node (as seen
        during breadth-first-search).

    Examples
    --------
    >>> G = eg.path_graph(4)
    >>> list(G)
    [0, 1, 2, 3]
    >>> eg.predecessor(G, 0)
    {0: [], 1: [0], 2: [1], 3: [2]}
    >>> eg.predecessor(G, 0, return_seen=True)
    ({0: [], 1: [0], 2: [1], 3: [2]}, {0: 0, 1: 1, 2: 2, 3: 3})


    """

    if source not in G:
        raise eg.NodeNotFound(f"Source {source} not in G")
    level = 0  # the current level
    nextlevel = [source]  # list of nodes to check at next level
    seen = {source: level}  # level (number of hops) when seen in BFS
    pred = {source: []}  # predecessor dictionary
    while nextlevel:
        level = level + 1
        thislevel = nextlevel
        nextlevel = []
        for v in thislevel:
            for w in list(G.neighbors(v)):
                if w not in seen:
                    pred[w] = [v]
                    seen[w] = level
                    nextlevel.append(w)
                elif seen[w] == level:  # add v to predecessor list if it
                    pred[w].append(v)  # is at the correct level
        if cutoff and cutoff <= level:
            break

    if target is not None:
        if return_seen:
            if target not in pred:
                return ([], -1)  # No predecessor
            return (pred[target], seen[target])
        else:
            if target not in pred:
                return []  # No predecessor
            return pred[target]
    else:
        if return_seen:
            return (pred, seen)
        else:
            return pred


# def main():
#     G = eg.path_graph(4)
#     print(G.edges)

#     print(predecessor(G, 0))


# if __name__ == "__main__":
#     main()
