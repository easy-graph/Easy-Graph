__all__ = ["get_structural_holes_HAM"]
from collections import Counter

import numpy as np

from easygraph.utils import *


eps = 2.220446049250313e-16


def sym(w):
    import scipy.linalg as spl

    """
    Initialize a random orthogonal matrix F = w * (wT * w)^ (-1/2)
    Parameters
    ----------
    w : A random matrix.

    Returns
    -------
    F : a random orthogonal matrix.
    """
    return w.dot(spl.inv(spl.sqrtm(w.T.dot(w))))


def avg_entropy(predicted_labels, actual_labels):
    """
    Calculate the average entropy between predicted_labels and actual_labels.

    Parameters
    ----------
    predicted_labels : a Ndarray of predicted_labels.
    actual_labels : a Ndarray of actual_labels.

    Returns
    -------
    A float of average entropy.
    """
    import scipy.stats as stat

    actual_labels_dict = {}
    predicted_labels_dict = {}
    for label in np.unique(actual_labels):
        actual_labels_dict[label] = np.nonzero(actual_labels == label)[0]
    for label in np.unique(predicted_labels):
        predicted_labels_dict[label] = np.nonzero(predicted_labels == label)[0]
    avg_value = 0
    N = len(predicted_labels)
    # store entropy for each community
    for label, items in predicted_labels_dict.items():
        N_i = float(len(items))
        p_i = []
        for label2, items2 in actual_labels_dict.items():
            common = set(items.tolist()).intersection(set(items2.tolist()))
            p_ij = float(len(common)) / N_i
            p_i.append(p_ij)
        entropy_i = stat.entropy(p_i)
        avg_value += entropy_i * (N_i / float(N))
    return avg_value


def load_adj_matrix(G):
    """
    Transfer the graph into sparse matrix.
    Parameters
    ----------
    G : graph
        An undirected graph.

    Returns
    -------
    A : A sparse matrix A
    """
    import scipy.sparse as sps

    listE = []
    for edge in G.edges:
        listE.append(edge[0] - 1)
        listE.append(edge[1] - 1)
    adj_tuples = np.array(listE).reshape(-1, 2)
    n = len(np.unique(adj_tuples))
    vals = np.array([1] * len(G.edges))
    max_id = max(max(adj_tuples[:, 0]), max(adj_tuples[:, 1])) + 1
    A = sps.csr_matrix(
        (vals, (adj_tuples[:, 0], adj_tuples[:, 1])), shape=(max_id, max_id)
    )
    A = A + A.T
    return sps.csr_matrix(A)


def majority_voting(votes):
    """
    majority voting.

    Parameters
    ----------
    votes : a Ndarray of votes

    Returns
    -------
    the most common label.
    """
    C = Counter(votes)
    pairs = C.most_common(2)
    if len(pairs) == 0:
        return 0
    if pairs[0][0] > 0:
        return pairs[0][0]
    elif len(pairs) > 1:
        return pairs[1][0]
    else:
        return 0


def label_by_neighbors(AdjMat, labels):
    """
    classifify SHS using majority voting.

    Parameters
    ----------
    AdjMat : adjacency matrix
    labels : a Ndarray of labeled communities of the nodes.

    Returns
    -------
    labels : a Ndarray of labeled communities of the nodes.
    """
    assert AdjMat.shape[0] == len(labels), "dimensions are not equal"
    unlabeled_idx = labels == 0
    num_unlabeled = sum(unlabeled_idx)
    count = 0
    while num_unlabeled > 0:
        idxs = np.array(np.nonzero(unlabeled_idx)[0])
        next_labels = np.zeros(len(labels))
        for idx in idxs:
            neighbors = np.nonzero(AdjMat[idx, :] > 0)[1]
            if len(neighbors) == 0:
                next_labels[idx] = majority_voting(labels)
            else:
                neighbor_labels = labels[neighbors]
                next_labels[idx] = majority_voting(neighbor_labels)
        labels[idxs] = next_labels[idxs]
        unlabeled_idx = labels == 0
        num_unlabeled = sum(unlabeled_idx)
    return labels


@not_implemented_for("multigraph")
def get_structural_holes_HAM(G, k, c, ground_truth_labels):
    """Structural hole spanners detection via HAM method.

    Using HAM [1]_ to jointly detect SHS and communities.

    Parameters
    ----------
    G : easygraph.Graph
        An undirected graph.

    k : int
        top - k structural hole spanners

    c : int
        the number of communities

    ground_truth_labels : list of lists
        The label of each node's community.

    Returns
    -------
    top_k_nodes : list
        The top-k structural hole spanners.

    SH_score : dict
        The structural hole spanners score for each node, given by HAM.

    cmnt_labels : dict
        The communities label of each node.


    Examples
    --------

    >>> get_structural_holes_HAM(G,
    ...                         k = 2, # To find top two structural holes spanners.
    ...                          c = 2,
    ...                          ground_truth_labels = [[0], [0], [1], [0], [1]] # The ground truth labels for each node - community detection result, for example.
    ...                         )

    References
    ----------
    .. [1] https://dl.acm.org/doi/10.1145/2939672.2939807

    """
    if k <= 0 or k > G.number_of_nodes():
        raise ValueError("`k` must be between 1 and number of nodes in the graph.")
    if c <= 0:
        raise ValueError("Number of communities `c` must be greater than 0")
    if len(ground_truth_labels) != G.number_of_nodes():
        raise ValueError(
            "Length of `ground_truth_labels` must match number of nodes in the graph."
        )
    import scipy.linalg as spl
    import scipy.sparse as sps

    from scipy.cluster.vq import kmeans
    from scipy.cluster.vq import vq
    from sklearn import metrics

    G_index, _, node_of_index = G.to_index_node_graph(begin_index=1)

    A_mat = load_adj_matrix(G_index)
    A = A_mat  # adjacency matrix
    n = A.shape[0]  # the number of nodes

    epsilon = 1e-4  # smoothing value: epsilon
    max_iter = 50  # maximum iteration value
    seeeed = 5433
    np.random.seed(seeeed)
    topk = k

    # Inv of degree matrix D^-1
    invD = sps.diags((np.array(A.sum(axis=0))[0, :] + eps) ** (-1.0), 0)
    # Laplacian matrix L = I - D^-1 * A
    L = (sps.identity(n) - invD.dot(A)).tocsr()
    # Initialize a random orthogonal matrix F
    F = sym(np.random.random((n, c)))

    # Algorithm 1
    for step in range(max_iter):
        Q = sps.identity(n).tocsr()
        P = L.dot(F)
        for i in range(n):
            Q[i, i] = 0.5 / (spl.norm(P[i, :]) + epsilon)

        R = L.T.dot(Q).dot(L)

        W, V = np.linalg.eigh(R.todense())
        Wsort = np.argsort(W)  # sort from smallest to largest
        F = V[:, Wsort[0:c]]  # select the smallest eigenvectors

    # find SH spanner
    SH = np.zeros((n,))
    for i in range(n):
        SH[i] = np.linalg.norm(F[i, :])
    SHrank = np.argsort(SH)  # index of SH

    # METRICS BEGIN

    to_keep_index = np.sort(SHrank[topk:])
    A_temp = A[to_keep_index, :]
    A_temp = A_temp[:, to_keep_index]
    HAM_labels_keep = np.asarray(ground_truth_labels)[to_keep_index]
    allLabels = np.asarray(ground_truth_labels)

    cluster_matrix = F
    labelbook, distortion = kmeans(cluster_matrix[to_keep_index, :], c)
    HAM_labels, dist = vq(cluster_matrix[to_keep_index, :], labelbook)

    print("AMI")
    print(
        "HAM: "
        + str(metrics.adjusted_mutual_info_score(HAM_labels, HAM_labels_keep.T[0]))
    )

    # classifify SHS using majority voting
    predLabels = np.zeros(len(ground_truth_labels))
    predLabels[to_keep_index] = HAM_labels + 1

    HAM_predLabels = label_by_neighbors(A, predLabels)
    print(
        "HAM_all: "
        + str(metrics.adjusted_mutual_info_score(HAM_predLabels, allLabels.T[0]))
    )

    print("NMI")
    print(
        "HAM: "
        + str(metrics.normalized_mutual_info_score(HAM_labels, HAM_labels_keep.T[0]))
    )
    print(
        "HAM_all: "
        + str(metrics.normalized_mutual_info_score(HAM_predLabels, allLabels.T[0]))
    )

    print("Entropy")
    print("HAM: " + str(avg_entropy(HAM_labels, HAM_labels_keep.T[0])))
    print("HAM_all: " + str(avg_entropy(HAM_predLabels, allLabels.T[0])))

    # METRICS END

    SH_score = dict()
    for index, rank in enumerate(SHrank):
        SH_score[node_of_index[index + 1]] = int(rank)

    cmnt_labels = dict()
    for index, label in enumerate(HAM_predLabels):
        cmnt_labels[node_of_index[index + 1]] = int(label)

    # top-k SHS
    top_k_ind = np.argpartition(SHrank, -k)[-k:]
    top_k_ind = top_k_ind[np.argsort(SHrank[top_k_ind])[::-1][:k]]
    top_k_nodes = []
    for ind in top_k_ind:
        top_k_nodes.append(node_of_index[ind + 1])

    return top_k_nodes, SH_score, cmnt_labels
