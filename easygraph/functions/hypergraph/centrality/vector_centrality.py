import easygraph as eg
import numpy as np

from easygraph.exception import EasyGraphError


__all__ = ["vector_centrality"]


def vector_centrality(H):
    """The vector centrality of nodes in the line graph of the hypergraph.

    Parameters
    ----------
    H : eg.Hypergraph


    Returns
    -------
    dict
        Centrality, where keys are node IDs and values are lists of centralities.

    References
    ----------
    "Vector centrality in hypergraphs", K. Kovalenko, M. Romance, E. Vasilyeva,
    D. Aleja, R. Criado, D. Musatov, A.M. Raigorodskii, J. Flores, I. Samoylenko,
    K. Alfaro-Bittner, M. Perc, S. Boccaletti,
    https://doi.org/10.1016/j.chaos.2022.112397

    """

    # If the hypergraph is empty, then return an empty dictionary
    if H.num_v == 0:
        return dict()

    LG = H.get_linegraph()
    if not eg.is_connected(LG):
        raise EasyGraphError("This method is not defined for disconnected hypergraphs.")
    LGcent = eg.eigenvector_centrality(LG)

    vc = {node: [] for node in range(0, H.num_v)}

    edge_label_dict = {tuple(edge): index for index, edge in enumerate(H.e[0])}

    hyperedge_dims = {tuple(edge): len(edge) for edge in H.e[0]}

    D = max([len(e) for e in H.e[0]])

    for k in range(2, D + 1):
        c_i = np.zeros(H.num_v)

        for edge, _ in list(filter(lambda x: x[1] == k, hyperedge_dims.items())):
            for node in edge:
                try:
                    c_i[node] += LGcent[edge_label_dict[edge]]
                except IndexError:
                    raise Exception(
                        "Nodes must be written with the Pythonic indexing (0,1,2...)"
                    )

        c_i *= 1 / k

        for node in range(H.num_v):
            vc[node].append(c_i[node])

    return vc
