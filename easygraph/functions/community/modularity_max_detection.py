from easygraph.functions.community.modularity import modularity
from easygraph.utils import *
from easygraph.utils.mapped_queue import MappedQueue


__all__ = ["greedy_modularity_communities"]


@not_implemented_for("multigraph")
def greedy_modularity_communities(G, weight="weight"):
    """Communities detection via greedy modularity method.

    Find communities in graph using Clauset-Newman-Moore greedy modularity
    maximization. This method currently supports the Graph class.

    Greedy modularity maximization begins with each node in its own community
    and joins the pair of communities that most increases modularity until no
    such pair exists.

    Parameters
    ----------
    G : easygraph.Graph or easygraph.DiGraph

    weight : string (default : 'weight')
        The key for edge weight. For undirected graph, it will regard each edge
        weight as 1.

    Returns
    ----------
    Yields sets of nodes, one for each community.

    References
    ----------
    .. [1] Newman, M. E. J. "Networks: An Introduction Oxford Univ." (2010).
    .. [2] Clauset, Aaron, Mark EJ Newman, and Cristopher Moore.
    "Finding community structure in very large networks." Physical review E 70.6 (2004): 066111.
    """

    # Count nodes and edges

    N = len(G.nodes)
    m = sum(d.get(weight, 1) for u, v, d in G.edges)
    if N == 0 or m == 0:
        print("Please input the graph which has at least one edge!")
        exit()
    q0 = 1.0 / (2.0 * m)

    # Map node labels to contiguous integers
    label_for_node = {i: v for i, v in enumerate(G.nodes)}
    node_for_label = {label_for_node[i]: i for i in range(N)}

    # Calculate degrees
    k_for_label = G.degree(weight=weight)
    k = [k_for_label[label_for_node[i]] for i in range(N)]

    # Initialize community and merge lists
    communities = {i: frozenset([i]) for i in range(N)}
    merges = []

    # Initial modularity
    partition = [[label_for_node[x] for x in c] for c in communities.values()]
    q_cnm = modularity(G, partition)

    # Initialize data structures
    # CNM Eq 8-9 (Eq 8 was missing a factor of 2 (from A_ij + A_ji)
    # a[i]: fraction of edges within community i
    # dq_dict[i][j]: dQ for merging community i, j
    # dq_heap[i][n] : (-dq, i, j) for communitiy i nth largest dQ
    # H[n]: (-dq, i, j) for community with nth largest max_j(dQ_ij)
    a = [k[i] * q0 for i in range(N)]
    dq_dict = {
        i: {
            j: 2 * q0 - 2 * k[i] * k[j] * q0 * q0
            for j in [node_for_label[u] for u in G.neighbors(label_for_node[i])]
            if j != i
        }
        for i in range(N)
    }
    dq_heap = [
        MappedQueue([(-dq, i, j) for j, dq in dq_dict[i].items()]) for i in range(N)
    ]
    H = MappedQueue([dq_heap[i].h[0] for i in range(N) if len(dq_heap[i]) > 0])

    # Merge communities until we can't improve modularity
    while len(H) > 1:
        # Find best merge
        # Remove from heap of row maxes
        # Ties will be broken by choosing the pair with lowest min community id
        try:
            dq, i, j = H.pop()
        except IndexError:
            break
        dq = -dq
        # Remove best merge from row i heap
        dq_heap[i].pop()
        # Push new row max onto H
        if len(dq_heap[i]) > 0:
            H.push(dq_heap[i].h[0])
        # If this element was also at the root of row j, we need to remove the
        # duplicate entry from H
        if dq_heap[j].h[0] == (-dq, j, i):
            H.remove((-dq, j, i))
            # Remove best merge from row j heap
            dq_heap[j].remove((-dq, j, i))
            # Push new row max onto H
            if len(dq_heap[j]) > 0:
                H.push(dq_heap[j].h[0])
        else:
            # Duplicate wasn't in H, just remove from row j heap
            dq_heap[j].remove((-dq, j, i))
        # Stop when change is non-positive
        if dq <= 0:
            break

        # Perform merge
        communities[j] = frozenset(communities[i] | communities[j])
        del communities[i]
        merges.append((i, j, dq))
        # New modularity
        q_cnm += dq
        # Get list of communities connected to merged communities
        i_set = set(dq_dict[i].keys())
        j_set = set(dq_dict[j].keys())
        all_set = (i_set | j_set) - {i, j}
        both_set = i_set & j_set
        # Merge i into j and update dQ
        for k in all_set:
            # Calculate new dq value
            if k in both_set:
                dq_jk = dq_dict[j][k] + dq_dict[i][k]
            elif k in j_set:
                dq_jk = dq_dict[j][k] - 2.0 * a[i] * a[k]
            else:
                # k in i_set
                dq_jk = dq_dict[i][k] - 2.0 * a[j] * a[k]
            # Update rows j and k
            for row, col in [(j, k), (k, j)]:
                # Save old value for finding heap index
                if k in j_set:
                    d_old = (-dq_dict[row][col], row, col)
                else:
                    d_old = None
                # Update dict for j,k only (i is removed below)
                dq_dict[row][col] = dq_jk
                # Save old max of per-row heap
                if len(dq_heap[row]) > 0:
                    d_oldmax = dq_heap[row].h[0]
                else:
                    d_oldmax = None
                # Add/update heaps
                d = (-dq_jk, row, col)
                if d_old is None:
                    # We're creating a new nonzero element, add to heap
                    dq_heap[row].push(d)
                else:
                    # Update existing element in per-row heap
                    dq_heap[row].update(d_old, d)
                # Update heap of row maxes if necessary
                if d_oldmax is None:
                    # No entries previously in this row, push new max
                    H.push(d)
                else:
                    # We've updated an entry in this row, has the max changed?
                    if dq_heap[row].h[0] != d_oldmax:
                        H.update(d_oldmax, dq_heap[row].h[0])

        # Remove row/col i from matrix
        i_neighbors = dq_dict[i].keys()
        for k in i_neighbors:
            # Remove from dict
            dq_old = dq_dict[k][i]
            del dq_dict[k][i]
            # Remove from heaps if we haven't already
            if k != j:
                # Remove both row and column
                for row, col in [(k, i), (i, k)]:
                    # Check if replaced dq is row max
                    d_old = (-dq_old, row, col)
                    if dq_heap[row].h[0] == d_old:
                        # Update per-row heap and heap of row maxes
                        dq_heap[row].remove(d_old)
                        H.remove(d_old)
                        # Update row max
                        if len(dq_heap[row]) > 0:
                            H.push(dq_heap[row].h[0])
                    else:
                        # Only update per-row heap
                        dq_heap[row].remove(d_old)

        del dq_dict[i]
        # Mark row i as deleted, but keep placeholder
        dq_heap[i] = MappedQueue()
        # Merge i into j and update a
        a[j] += a[i]
        a[i] = 0

    communities = [
        frozenset(label_for_node[i] for i in c) for c in communities.values()
    ]
    return sorted(communities, key=len, reverse=True)
