import easygraph as eg

from easygraph.utils.exception import EasyGraphError


__all__ = [
    "random_position",
    "circular_position",
    "shell_position",
    "rescale_position",
    "kamada_kawai_layout",
    # "spring_layout",
    # "fruchterman_reingold_layout",
    # "_process_params",
    # "_fruchterman_reingold",
    # "_sparse_fruchterman_reingold",
]


def random_position(G, center=None, dim=2, random_seed=None):
    """
    Returns random position for each node in graph G.

    Parameters
    ----------
    G : easygraph.Graph or easygraph.DiGraph

    center : array-like or None, optional (default : None)
        Coordinate pair around which to center the layout

    dim : int, optional (default : 2)
        Dimension of layout

    random_seed : int or None, optional (default : None)
        Seed for RandomState instance

    Returns
    ----------
    pos : dict
        A dictionary of positions keyed by node
    """
    import numpy as np

    center = _get_center(center, dim)

    rng = np.random.RandomState(seed=random_seed)
    pos = rng.rand(len(G), dim) + center
    pos = pos.astype(np.float32)
    pos = dict(zip(G, pos))

    return pos


def circular_position(G, center=None, scale=1):
    """
    Position nodes on a circle, the dimension is 2.

    Parameters
    ----------
    G : easygraph.Graph or easygraph.DiGraph
        A position will be assigned to every node in G

    center : array-like or None, optional (default : None)
        Coordinate pair around which to center the layout

    scale : number, optional (default : 1)
        Scale factor for positions

    Returns
    -------
    pos : dict
        A dictionary of positions keyed by node
    """
    import numpy as np

    center = _get_center(center, dim=2)

    if len(G) == 0:
        pos = {}
    elif len(G) == 1:
        pos = {G.nodes[0]: center}
    else:
        theta = np.linspace(0, 1, len(G), endpoint=False) * 2 * np.pi
        theta = theta.astype(np.float32)
        pos = np.column_stack([np.cos(theta), np.sin(theta)])
        pos = rescale_position(pos, scale=scale) + center
        pos = dict(zip(G, pos))

    return pos


def shell_position(G, nlist=None, scale=1, center=None):
    """
    Position nodes in concentric circles, the dimension is 2.

    Parameters
    ----------
    G : easygraph.Graph or easygraph.DiGraph

    nlist : list of lists or None, optional (default : None)
       List of node lists for each shell.

    scale : number, optional (default : 1)
        Scale factor for positions.

    center : array-like or None, optional (default : None)
        Coordinate pair around which to center the layout.


    Returns
    -------
    pos : dict
        A dictionary of positions keyed by node

    Notes
    -----
    This algorithm currently only works in two dimensions and does not
    try to minimize edge crossings.

    """
    import numpy as np

    center = _get_center(center, dim=2)

    if len(G) == 0:
        return {}
    if len(G) == 1:
        return {G.nodes[0]: center}

    if nlist is None:
        # draw the whole graph in one shell
        nlist = [list(G)]

    if len(nlist[0]) == 1:
        # single node at center
        radius = 0.0
    else:
        # else start at r=1
        radius = 1.0

    npos = {}
    for nodes in nlist:
        # Discard the extra angle since it matches 0 radians.
        theta = np.linspace(0, 1, len(nodes), endpoint=False) * 2 * np.pi
        theta = theta.astype(np.float32)
        pos = np.column_stack([np.cos(theta), np.sin(theta)])
        if len(pos) > 1:
            pos = rescale_position(pos, scale=scale * radius / len(nlist)) + center
        else:
            pos = np.array([(scale * radius + center[0], center[1])])
        npos.update(zip(nodes, pos))
        radius += 1.0

    return npos


def _get_center(center, dim):
    import numpy as np

    if center is None:
        center = np.zeros(dim)
    else:
        center = np.asarray(center)

    if dim < 2:
        raise ValueError("cannot handle dimensions < 2")

    if len(center) != dim:
        msg = "length of center coordinates must match dimension of layout"
        raise ValueError(msg)

    return center


def rescale_position(pos, scale=1):
    """
    Returns scaled position array to (-scale, scale) in all axes.

    Parameters
    ----------
    pos : numpy array
        positions to be scaled. Each row is a position.

    scale : number, optional (default : 1)
        The size of the resulting extent in all directions.

    Returns
    -------
    pos : numpy array
        scaled positions. Each row is a position.
    """
    # Find max length over all dimensions
    assert (
        len(pos.shape) != 1
    ), "One-dimensional ndarray is not available for rescaling."
    lim = 0  # max coordinate for all axes
    for i in range(pos.shape[1]):
        pos[:, i] -= pos[:, i].mean()
        lim = max(abs(pos[:, i]).max(), lim)
    # rescale to (-scale, scale) in all directions, preserves aspect
    if lim > 0:
        for i in range(pos.shape[1]):
            pos[:, i] *= scale / lim
    return pos


def kamada_kawai_layout(
    G, dist=None, pos=None, weight="weight", scale=1, center=None, dim=2
):
    """Position nodes using Kamada-Kawai basic-length cost-function.

    Parameters
    ----------
    G : graph or list of nodes
        A position will be assigned to every node in G.

    dist : dict (default=None)
        A two-level dictionary of optimal distances between nodes,
        indexed by source and destination node.
        If None, the distance is computed using shortest_path_length().

    pos : dict or None  optional (default=None)
        Initial positions for nodes as a dictionary with node as keys
        and values as a coordinate list or tuple.  If None, then use
        circular_layout() for dim >= 2 and a linear layout for dim == 1.

    weight : string or None   optional (default='weight')
        The edge attribute that holds the numerical value used for
        the edge weight.  If None, then all edge weights are 1.

    scale : number (default: 1)
        Scale factor for positions.

    center : array-like or None
        Coordinate pair around which to center the layout.

    dim : int
        Dimension of layout.

    Returns
    -------
    pos : dict
        A dictionary of positions keyed by node

    Examples
    --------
    >>> pos = eg.kamada_kawai_layout(G)
    """
    import numpy as np

    nNodes = len(G)
    if nNodes == 0:
        return {}

    if dist is None:
        dist = dict(eg.Floyd(G))
    dist_mtx = 1e6 * np.ones((nNodes, nNodes))
    for row, nr in enumerate(G):
        if nr not in dist:
            continue
        rdist = dist[nr]
        for col, nc in enumerate(G):
            if nc not in rdist:
                continue
            dist_mtx[row][col] = rdist[nc]

    if pos is None:
        if dim >= 3:
            pos = eg.random_position(G, dim=dim)
        elif dim == 2:
            pos = eg.circular_position(G)
        else:
            pos = {n: pt for n, pt in zip(G, np.linspace(0, 1, len(G)))}

    pos_arr = np.array([pos[n] for n in G])

    pos = _kamada_kawai_solve(dist_mtx, pos_arr, dim)

    if center is None:
        center = np.zeros(dim)
    else:
        center = np.asarray(center)

    if len(center) != dim:
        msg = "length of center coordinates must match dimension of layout"
        raise ValueError(msg)

    pos = eg.rescale_position(pos, scale=scale) + center
    return dict(zip(G, pos))


def _kamada_kawai_solve(dist_mtx, pos_arr, dim):
    # Anneal node locations based on the Kamada-Kawai cost-function,
    # using the supplied matrix of preferred inter-node distances,
    # and starting locations.

    import numpy as np

    from scipy.optimize import minimize

    meanwt = 1e-3
    costargs = (np, 1 / (dist_mtx + np.eye(dist_mtx.shape[0]) * 1e-3), meanwt, dim)

    optresult = minimize(
        _kamada_kawai_costfn,
        pos_arr.ravel(),
        method="L-BFGS-B",
        args=costargs,
        jac=True,
    )

    return optresult.x.reshape((-1, dim))


def _kamada_kawai_costfn(pos_vec, np, invdist, meanweight, dim):
    # Cost-function and gradient for Kamada-Kawai layout algorithm
    nNodes = invdist.shape[0]
    pos_arr = pos_vec.reshape((nNodes, dim))

    delta = pos_arr[:, np.newaxis, :] - pos_arr[np.newaxis, :, :]
    nodesep = np.linalg.norm(delta, axis=-1)
    direction = np.einsum("ijk,ij->ijk", delta, 1 / (nodesep + np.eye(nNodes) * 1e-3))

    offset = nodesep * invdist - 1.0
    offset[np.diag_indices(nNodes)] = 0

    cost = 0.5 * np.sum(offset**2)
    grad = np.einsum("ij,ij,ijk->ik", invdist, offset, direction) - np.einsum(
        "ij,ij,ijk->jk", invdist, offset, direction
    )

    # Additional parabolic term to encourage mean position to be near origin:
    sumpos = np.sum(pos_arr, axis=0)
    cost += 0.5 * meanweight * np.sum(sumpos**2)
    grad += meanweight * sumpos

    return (cost, grad.ravel())


# @np_random_state(10)
# def spring_layout(
#     G,
#     k=None,
#     pos=None,
#     fixed=None,
#     iterations=50,
#     threshold=1e-4,
#     weight="weight",
#     scale=1,
#     center=None,
#     dim=2,
#     seed=None,
# ):
#     """Position nodes using Fruchterman-Reingold force-directed algorithm.
#
#     The algorithm simulates a force-directed representation of the network
#     treating edges as springs holding nodes close, while treating nodes
#     as repelling objects, sometimes called an anti-gravity force.
#     Simulation continues until the positions are close to an equilibrium.
#
#     There are some hard-coded values: minimal distance between
#     nodes (0.01) and "temperature" of 0.1 to ensure nodes don't fly away.
#     During the simulation, `k` helps determine the distance between nodes,
#     though `scale` and `center` determine the size and place after
#     rescaling occurs at the end of the simulation.
#
#     Fixing some nodes doesn't allow them to move in the simulation.
#     It also turns off the rescaling feature at the simulation's end.
#     In addition, setting `scale` to `None` turns off rescaling.
#
#     Parameters
#     ----------
#     G : EasyGraph graph or list of nodes
#         A position will be assigned to every node in G.
#
#     k : float (default=None)
#         Optimal distance between nodes.  If None the distance is set to
#         1/sqrt(n) where n is the number of nodes.  Increase this value
#         to move nodes farther apart.
#
#     pos : dict or None  optional (default=None)
#         Initial positions for nodes as a dictionary with node as keys
#         and values as a coordinate list or tuple.  If None, then use
#         random initial positions.
#
#     fixed : list or None  optional (default=None)
#         Nodes to keep fixed at initial position.
#         Nodes not in ``G.nodes`` are ignored.
#         ValueError raised if `fixed` specified and `pos` not.
#
#     iterations : int  optional (default=50)
#         Maximum number of iterations taken
#
#     threshold: float optional (default = 1e-4)
#         Threshold for relative error in node position changes.
#         The iteration stops if the error is below this threshold.
#
#     weight : string or None   optional (default='weight')
#         The edge attribute that holds the numerical value used for
#         the edge weight.  Larger means a stronger attractive force.
#         If None, then all edge weights are 1.
#
#     scale : number or None (default: 1)
#         Scale factor for positions. Not used unless `fixed is None`.
#         If scale is None, no rescaling is performed.
#
#     center : array-like or None
#         Coordinate pair around which to center the layout.
#         Not used unless `fixed is None`.
#
#     dim : int
#         Dimension of layout.
#
#     seed : int, RandomState instance or None  optional (default=None)
#         Set the random state for deterministic node layouts.
#         If int, `seed` is the seed used by the random number generator,
#         if numpy.random.RandomState instance, `seed` is the random
#         number generator,
#         if None, the random number generator is the RandomState instance used
#         by numpy.random.
#
#     Returns
#     -------
#     pos : dict
#         A dictionary of positions keyed by node
#
#     Examples
#     --------
#     >>> G = eg.path_graph(4)
#     >>> pos = eg.spring_layout(G)
#
#
#     """
#     import numpy as np
#
#     G, center = _process_params(G, center, dim)
#
#     if fixed is not None:
#         if pos is None:
#             raise ValueError("nodes are fixed without positions given")
#         for node in fixed:
#             if node not in pos:
#                 raise ValueError("nodes are fixed without positions given")
#         nfixed = {node: i for i, node in enumerate(G)}
#         fixed = np.asarray([nfixed[node] for node in fixed if node in nfixed])
#
#     if pos is not None:
#         # Determine size of existing domain to adjust initial positions
#         dom_size = max(coord for pos_tup in pos.values() for coord in pos_tup)
#         if dom_size == 0:
#             dom_size = 1
#         pos_arr = seed.rand(len(G), dim) * dom_size + center
#
#         for i, n in enumerate(G):
#             if n in pos:
#                 pos_arr[i] = np.asarray(pos[n])
#     else:
#         pos_arr = None
#         dom_size = 1
#
#     if len(G) == 0:
#         return {}
#     if len(G) == 1:
#         return {eg.utils.arbitrary_element(G.nodes()): center}
#
#     try:
#         # Sparse matrix
#         if len(G) < 500:  # sparse solver for large graphs
#             raise ValueError
#         A = eg.to_scipy_sparse_array(G, weight=weight, dtype="f")
#         if k is None and fixed is not None:
#             # We must adjust k by domain size for layouts not near 1x1
#             nnodes, _ = A.shape
#             k = dom_size / np.sqrt(nnodes)
#         pos = _sparse_fruchterman_reingold(
#             A, k, pos_arr, fixed, iterations, threshold, dim, seed
#         )
#     except ValueError:
#         A = eg.to_numpy_array(G, weight=weight)
#         if k is None and fixed is not None:
#             # We must adjust k by domain size for layouts not near 1x1
#             nnodes, _ = A.shape
#             k = dom_size / np.sqrt(nnodes)
#         pos = _fruchterman_reingold(
#             A, k, pos_arr, fixed, iterations, threshold, dim, seed
#         )
#     if fixed is None and scale is not None:
#         pos = rescale_position(pos, scale=scale) + center
#     pos = dict(zip(G, pos))
#     return pos
#
# fruchterman_reingold_layout = spring_layout
#
# def _process_params(G, center, dim):
#     # Some boilerplate code.
#     import numpy as np
#
#     if not isinstance(G, eg.Graph):
#         empty_graph = eg.Graph()
#         empty_graph.add_nodes_from(G)
#         G = empty_graph
#
#     if center is None:
#         center = np.zeros(dim)
#     else:
#         center = np.asarray(center)
#
#     if len(center) != dim:
#         msg = "length of center coordinates must match dimension of layout"
#         raise ValueError(msg)
#
#     return G, center
#
# @np_random_state(7)
# def _fruchterman_reingold(
#     A, k=None, pos=None, fixed=None, iterations=50, threshold=1e-4, dim=2, seed=None
# ):
#     # Position nodes in adjacency matrix A using Fruchterman-Reingold
#     # Entry point for NetworkX graph is fruchterman_reingold_layout()
#     import numpy as np
#
#     try:
#         nnodes, _ = A.shape
#     except AttributeError as err:
#         msg = "fruchterman_reingold() takes an adjacency matrix as input"
#         raise EasyGraphError(msg) from err
#
#     if pos is None:
#         # random initial positions
#         pos = np.asarray(seed.rand(nnodes, dim), dtype=A.dtype)
#     else:
#         # make sure positions are of same type as matrix
#         pos = pos.astype(A.dtype)
#
#     # optimal distance between nodes
#     if k is None:
#         k = np.sqrt(1.0 / nnodes)
#     # the initial "temperature"  is about .1 of domain area (=1x1)
#     # this is the largest step allowed in the dynamics.
#     # We need to calculate this in case our fixed positions force our domain
#     # to be much bigger than 1x1
#     t = max(max(pos.T[0]) - min(pos.T[0]), max(pos.T[1]) - min(pos.T[1])) * 0.1
#     # simple cooling scheme.
#     # linearly step down by dt on each iteration so last iteration is size dt.
#     dt = t / (iterations + 1)
#     delta = np.zeros((pos.shape[0], pos.shape[0], pos.shape[1]), dtype=A.dtype)
#     # the inscrutable (but fast) version
#     # this is still O(V^2)
#     # could use multilevel methods to speed this up significantly
#     for iteration in range(iterations):
#         # matrix of difference between points
#         delta = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
#         # distance between points
#         distance = np.linalg.norm(delta, axis=-1)
#         # enforce minimum distance of 0.01
#         np.clip(distance, 0.01, None, out=distance)
#         # displacement "force"
#         displacement = np.einsum(
#             "ijk,ij->ik", delta, (k * k / distance**2 - A * distance / k)
#         )
#         # update positions
#         length = np.linalg.norm(displacement, axis=-1)
#         length = np.where(length < 0.01, 0.1, length)
#         delta_pos = np.einsum("ij,i->ij", displacement, t / length)
#         if fixed is not None:
#             # don't change positions of fixed nodes
#             delta_pos[fixed] = 0.0
#         pos += delta_pos
#         # cool temperature
#         t -= dt
#         if (np.linalg.norm(delta_pos) / nnodes) < threshold:
#             break
#     return pos
#
# @np_random_state(7)
# def _sparse_fruchterman_reingold(
#     A, k=None, pos=None, fixed=None, iterations=50, threshold=1e-4, dim=2, seed=None
# ):
#     # Position nodes in adjacency matrix A using Fruchterman-Reingold
#     # Entry point for NetworkX graph is fruchterman_reingold_layout()
#     # Sparse version
#     import numpy as np
#     import scipy as sp
#     import scipy.sparse  # call as sp.sparse
#
#     try:
#         nnodes, _ = A.shape
#     except AttributeError as err:
#         msg = "fruchterman_reingold() takes an adjacency matrix as input"
#         raise EasyGraphError(msg) from err
#     # make sure we have a LIst of Lists representation
#     try:
#         A = A.tolil()
#     except AttributeError:
#         A = (sp.sparse.coo_array(A)).tolil()
#
#     if pos is None:
#         # random initial positions
#         pos = np.asarray(seed.rand(nnodes, dim), dtype=A.dtype)
#     else:
#         # make sure positions are of same type as matrix
#         pos = pos.astype(A.dtype)
#
#     # no fixed nodes
#     if fixed is None:
#         fixed = []
#
#     # optimal distance between nodes
#     if k is None:
#         k = np.sqrt(1.0 / nnodes)
#     # the initial "temperature"  is about .1 of domain area (=1x1)
#     # this is the largest step allowed in the dynamics.
#     t = max(max(pos.T[0]) - min(pos.T[0]), max(pos.T[1]) - min(pos.T[1])) * 0.1
#     # simple cooling scheme.
#     # linearly step down by dt on each iteration so last iteration is size dt.
#     dt = t / (iterations + 1)
#
#     displacement = np.zeros((dim, nnodes))
#     for iteration in range(iterations):
#         displacement *= 0
#         # loop over rows
#         for i in range(A.shape[0]):
#             if i in fixed:
#                 continue
#             # difference between this row's node position and all others
#             delta = (pos[i] - pos).T
#             # distance between points
#             distance = np.sqrt((delta**2).sum(axis=0))
#             # enforce minimum distance of 0.01
#             distance = np.where(distance < 0.01, 0.01, distance)
#             # the adjacency matrix row
#             Ai = A.getrowview(i).toarray()  # TODO: revisit w/ sparse 1D container
#             # displacement "force"
#             displacement[:, i] += (
#                 delta * (k * k / distance**2 - Ai * distance / k)
#             ).sum(axis=1)
#         # update positions
#         length = np.sqrt((displacement**2).sum(axis=0))
#         length = np.where(length < 0.01, 0.1, length)
#         delta_pos = (displacement * t / length).T
#         pos += delta_pos
#         # cool temperature
#         t -= dt
#         if (np.linalg.norm(delta_pos) / nnodes) < threshold:
#             break
#     return pos
