import easygraph as eg



__all__ = [
    "random_position",
    "circular_position",
    "shell_position",
    "rescale_position"
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
