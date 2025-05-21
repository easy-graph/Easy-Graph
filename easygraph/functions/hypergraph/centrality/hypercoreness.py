from itertools import compress

import easygraph as eg
import numpy as np


__all__ = ["size_independent_hypercoreness", "frequency_based_hypercoreness"]


def size_independent_hypercoreness(h):
    """The size_independent_hypercoreness of nodes in hypergraph.

    Parameters
    ----------
    h : eg.Hypergraph.


    Returns
    ----------
    dict
        Centrality, where keys are node IDs and values are lists of centralities.

    References
    ----------
    Mancastroppa, M., Iacopini, I., Petri, G. et al. Hyper-cores promote localization and efficient seeding in higher-order processes. Nat Commun 14, 6223 (2023). https://doi.org/10.1038/s41467-023-41887-2.

    """
    e_list = h.e[0]
    initial_node_num = h.num_v
    data = [e_list[i] for i in range(len(e_list)) if len(e_list[i]) > 1]

    data.sort(key=len)
    L = len(data)
    size_max = len(data[L - 1])

    size = list([len(data[j]) for j in range(L)])

    X = eg.Hypergraph(num_v=initial_node_num, e_list=data)
    IDX = list(range(0, X.num_v))

    M = range(2, size_max + 1)
    k_step = 1
    K = range(1, 1200, k_step)
    k_shell_dict = {}
    idx_orig = IDX

    IDX_size = range(len(size))
    k_max = np.zeros(len(M))

    for j in idx_orig:
        k_shell_dict[j] = np.zeros(len(M))

    for x in range(len(M)):
        m = M[x]

        D = np.zeros(len(K))

        # consider only hyperedges of size >=m
        idx_size = list(
            compress(IDX_size, np.greater_equal(size, m * np.ones(len(size))))
        )
        int_sel = list([data[i] for i in idx_size])
        # build hypergraph with only interactions of size >=m
        X = eg.Hypergraph(num_v=initial_node_num, e_list=int_sel)
        node_set = set()
        for sublist in int_sel:
            for element in sublist:
                node_set.add(element)
        IDX = list(node_set)
        # IDX_e = list(X.e[0])

        for y in range(len(K)):
            kk = K[y]

            d_tot_m = np.zeros(len(IDX))
            prev_shell = IDX

            for i in range(len(IDX)):
                d_tot_m[i] = X.degree_node[IDX[i]]

            idx_n_remove = list(
                compress(IDX, np.greater(kk * np.ones(len(d_tot_m)), d_tot_m))
            )  # nodes with degree<k are removed
            # X.remove_nodes_from(idx_n_remove)
            now_e_list = X.e[0]
            new_e_list = []
            for e in now_e_list:
                new_e = []
                for n in e:
                    if n not in idx_n_remove:
                        new_e.append(n)
                if len(new_e) > 0:
                    new_e_list.append(new_e)

            X = eg.Hypergraph(num_v=initial_node_num, e_list=new_e_list)

            IDX_e = list(range(0, len(X.e[0])))

            sizes = [
                len(X.e[0][i]) for i in IDX_e
            ]  # hyperedges with size <m are removed
            idx_e_remove = [IDX_e[i] for i in range(len(IDX_e)) if sizes[i] < m]
            now_e_list = X.e[0]
            new_e_list = []
            for i in range(len(now_e_list)):
                if i not in idx_e_remove:
                    new_e_list.append(now_e_list[i])

            X = eg.Hypergraph(num_v=initial_node_num, e_list=new_e_list)

            node_set = set()
            for sublist in X.e[0]:
                for element in sublist:
                    node_set.add(element)
            IDX = list(node_set)

            while len(idx_n_remove) > 0 or len(idx_e_remove) > 0:
                d_tot_m = np.zeros(len(IDX))

                for i in range(len(IDX)):
                    d_tot_m[i] = X.degree_node[IDX[i]]

                idx_n_remove = list(
                    compress(IDX, np.greater(kk * np.ones(len(d_tot_m)), d_tot_m))
                )  # nodes with degree<k are removed
                # X.remove_nodes_from(idx_n_remove)
                now_e_list = X.e[0]
                new_e_list = []
                for e in now_e_list:
                    new_e = []
                    for n in e:
                        if n not in idx_n_remove:
                            new_e.append(n)
                    if len(new_e) > 0:
                        new_e_list.append(new_e)
                X = eg.Hypergraph(num_v=initial_node_num, e_list=new_e_list)

                IDX_e = list(range(len(X.e[0])))
                sizes = [
                    len(X.e[0][i]) for i in IDX_e
                ]  # hyperedges with size <m are removed

                idx_e_remove = [IDX_e[i] for i in range(len(IDX_e)) if sizes[i] < m]
                now_e_list = X.e[0]
                new_e_list = []
                for i in range(len(now_e_list)):
                    if i not in idx_e_remove:
                        new_e_list.append(now_e_list[i])
                X = eg.Hypergraph(num_v=initial_node_num, e_list=new_e_list)

                node_set = set()
                for sublist in X.e[0]:
                    for element in sublist:
                        node_set.add(element)
                IDX = list(node_set)

            shell_kk = list(sorted(set(prev_shell) - set(IDX)))
            for j in shell_kk:
                # if j not in idx_n_remove:
                #     continue
                k_shell_dict[j][x] = kk - k_step

            node_set = set()
            for sublist in X.e[0]:
                for element in sublist:
                    node_set.add(element)
            IDX = list(node_set)

            D[y] = len(node_set)
            if y > 0:
                if D[y] == 0 and D[y - 1] != 0:
                    # maximum connectivity at order m
                    k_max[x] = kk - k_step
            # stop the decomposition when the (k,m)-core is empty
            if D[y] == 0:
                break

    # size-independent hypercoreness
    R_dict = {}
    for y in k_shell_dict:
        R_dict[y] = sum(np.array(k_shell_dict[y]) / np.array(k_max))

    return R_dict


def frequency_based_hypercoreness(h):
    r"""The frequency-based hypercoreness of nodes in hypergraph.

     Parameters
     ----------
     h : easygraph.Hypergraph


    Returns
    -------
    dict : Centrality, where keys are node IDs and values are lists of centralities.

    References
    ----------
    Mancastroppa, M., Iacopini, I., Petri, G. et al. Hyper-cores promote localization and efficient seeding in higher-order processes. Nat Commun 14, 6223 (2023). https://doi.org/10.1038/s41467-023-41887-2

    """
    e_list = h.e[0]
    initial_node_num = h.num_v
    data = [e_list[i] for i in range(len(e_list)) if len(e_list[i]) > 1]

    data.sort(key=len)
    L = len(data)
    size_max = len(data[L - 1])

    size = list([len(data[j]) for j in range(L)])

    X = eg.Hypergraph(num_v=initial_node_num, e_list=data)
    IDX = list(range(0, X.num_v))

    M = range(2, size_max + 1)
    k_step = 1
    K = range(1, 1200, k_step)
    k_shell_dict = {}
    idx_orig = IDX

    IDX_size = range(len(size))
    k_max = np.zeros(len(M))

    for j in idx_orig:
        k_shell_dict[j] = np.zeros(len(M))

    for x in range(len(M)):
        m = M[x]

        D = np.zeros(len(K))
        # consider only hyperedges of size >=m
        idx_size = list(
            compress(IDX_size, np.greater_equal(size, m * np.ones(len(size))))
        )
        int_sel = list([data[i] for i in idx_size])
        # build hypergraph with only interactions of size >=m
        X = eg.Hypergraph(num_v=initial_node_num, e_list=int_sel)
        node_set = set()
        for sublist in int_sel:
            for element in sublist:
                node_set.add(element)
        IDX = list(node_set)

        for y in range(len(K)):
            kk = K[y]

            d_tot_m = np.zeros(len(IDX))
            prev_shell = IDX

            for i in range(len(IDX)):
                d_tot_m[i] = X.degree_node[IDX[i]]

            idx_n_remove = list(
                compress(IDX, np.greater(kk * np.ones(len(d_tot_m)), d_tot_m))
            )  # nodes with degree<k are removed
            now_e_list = X.e[0]
            new_e_list = []
            for e in now_e_list:
                new_e = []
                for n in e:
                    if n not in idx_n_remove:
                        new_e.append(n)
                if len(new_e) > 0:
                    new_e_list.append(new_e)

            X = eg.Hypergraph(num_v=initial_node_num, e_list=new_e_list)

            IDX_e = list(range(0, len(X.e[0])))

            # hyperedges with size <m are removed
            sizes = [len(X.e[0][i]) for i in IDX_e]
            idx_e_remove = [IDX_e[i] for i in range(len(IDX_e)) if sizes[i] < m]
            now_e_list = X.e[0]
            new_e_list = []
            for i in range(len(now_e_list)):
                if i not in idx_e_remove:
                    new_e_list.append(now_e_list[i])

            X = eg.Hypergraph(num_v=initial_node_num, e_list=new_e_list)

            node_set = set()
            for sublist in X.e[0]:
                for element in sublist:
                    node_set.add(element)
            IDX = list(node_set)

            while len(idx_n_remove) > 0 or len(idx_e_remove) > 0:
                d_tot_m = np.zeros(len(IDX))

                for i in range(len(IDX)):
                    d_tot_m[i] = X.degree_node[IDX[i]]
                # nodes with degree<k are removed
                idx_n_remove = list(
                    compress(IDX, np.greater(kk * np.ones(len(d_tot_m)), d_tot_m))
                )
                now_e_list = X.e[0]
                new_e_list = []
                for e in now_e_list:
                    new_e = []
                    for n in e:
                        if n not in idx_n_remove:
                            new_e.append(n)
                    if len(new_e) > 0:
                        new_e_list.append(new_e)
                X = eg.Hypergraph(num_v=initial_node_num, e_list=new_e_list)

                IDX_e = list(range(len(X.e[0])))
                # hyperedges with size <m are removed
                sizes = [len(X.e[0][i]) for i in IDX_e]

                idx_e_remove = [IDX_e[i] for i in range(len(IDX_e)) if sizes[i] < m]
                now_e_list = X.e[0]
                new_e_list = []
                for i in range(len(now_e_list)):
                    if i not in idx_e_remove:
                        new_e_list.append(now_e_list[i])
                X = eg.Hypergraph(num_v=initial_node_num, e_list=new_e_list)

                node_set = set()
                for sublist in X.e[0]:
                    for element in sublist:
                        node_set.add(element)
                IDX = list(node_set)

            shell_kk = list(sorted(set(prev_shell) - set(IDX)))
            for j in shell_kk:
                k_shell_dict[j][x] = kk - k_step

            node_set = set()
            for sublist in X.e[0]:
                for element in sublist:
                    node_set.add(element)
            IDX = list(node_set)

            D[y] = len(node_set)
            if y > 0:
                if D[y] == 0 and D[y - 1] != 0:
                    k_max[x] = kk - k_step  # maximum connectivity at order m
            if D[y] == 0:
                break  # stop the decomposition when the (k,m)-core is empty

    # Psi(m)  distribution of hyperedges size
    Psi = []
    for m in range(2, size_max + 1):
        Psi.append(size.count(m) / len(size))
    # frequency-based hypercoreness
    R_w_dict = {}
    for y in k_shell_dict:
        R_w_dict[y] = sum(np.array(Psi) * np.array(k_shell_dict[y]) / np.array(k_max))
    return R_w_dict
