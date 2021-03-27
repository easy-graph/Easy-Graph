import easygraph as eg
import numpy as np


__all__ = [
    "random_position",
    "circular_position",
    "shell_position",
    "rescale_position",
    "kamada_kawai_layout"
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

    random_seed : int or None, optinal (default : None)
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
        raise ValueError('cannot handle dimensions < 2')

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
    """Position nodes using Kamada-Kawai path-length cost-function.

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
            pos = eg.random_position(G,dim=dim)
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

    cost = 0.5 * np.sum(offset ** 2)
    grad = np.einsum("ij,ij,ijk->ik", invdist, offset, direction) - np.einsum(
        "ij,ij,ijk->jk", invdist, offset, direction
    )

    # Additional parabolic term to encourage mean position to be near origin:
    sumpos = np.sum(pos_arr, axis=0)
    cost += 0.5 * meanweight * np.sum(sumpos ** 2)
    grad += meanweight * sumpos

    return (cost, grad.ravel())
