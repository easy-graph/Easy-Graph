import random
import sys

import easygraph as eg
import numpy as np
import pytest

from easygraph.functions.basic.localassort import localAssort


class TestLocalAssort:
    @classmethod
    def setup_class(self):
        self.G = eg.get_graph_karateclub()
        random_value = [0, 1, 2, 3, 4, 5]
        edgelist = []
        valuelist = []
        node_num = len(self.G.nodes)
        for e in self.G.edges:
            edgelist.append([e[0] - 1, e[1] - 1])
        for i in range(0, node_num):
            valuelist.append(random.choice(random_value))
        self.edgelist = np.int32(edgelist)
        valuelist = np.int32(valuelist)
        self.valuelist = valuelist

    @pytest.mark.skipif(
        sys.version_info.major <= 3 and sys.version_info.minor <= 7,
        reason="python version should higher than 3.7",
    )
    def test_karateclub(self):
        assortM, assortT, Z = eg.localAssort(
            self.edgelist, self.valuelist, pr=np.arange(0, 1, 0.1)
        )

        _, assortT, Z = eg.functions.basic.localassort.localAssort(
            self.edgelist, self.valuelist, pr=np.array([0.9])
        )


def test_localassort_small_complete_graph():
    G = eg.complete_graph(4)
    edgelist = np.array(list(G.edges))
    node_attr = np.array([0, 0, 1, 1])
    assortM, assortT, Z = localAssort(edgelist, node_attr)
    assert assortM.shape == (4, 10)
    assert assortT.shape == (4,)
    assert Z.shape == (4,)
    assert np.all(Z >= 0) and np.all(Z <= 1)


def test_localassort_with_missing_attributes():
    G = eg.path_graph(5)
    edgelist = np.array(list(G.edges))
    node_attr = np.array([0, -1, 1, -1, 1])
    assortM, assortT, Z = localAssort(edgelist, node_attr, pr=np.array([0.5]))
    assert assortT.shape == (5,)
    assert Z.shape == (5,)
    assert np.any(np.isnan(assortT))


def test_localassort_directed_graph():
    G = eg.DiGraph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3)])
    edgelist = np.array(list(G.edges))
    node_attr = np.array([0, 1, 0, 1])
    assortM, assortT, Z = localAssort(edgelist, node_attr, undir=False)
    assert assortM.shape == (4, 10)
    assert assortT.shape == (4,)
    assert Z.shape == (4,)


def test_localassort_single_node_graph():
    edgelist = np.empty((0, 2), dtype=int)
    node_attr = np.array([0])
    assortM, assortT, Z = localAssort(edgelist, node_attr)
    assert assortM.shape == (1, 10)
    assert np.all(np.isnan(assortM)) or np.allclose(assortM, 0, atol=1e-5)
    assert np.all(np.isnan(assortT)) or np.allclose(assortT, 0, atol=1e-5)
    assert np.all(np.isnan(Z)) or np.allclose(Z, 0, atol=1e-5)


def test_localassort_disconnected_graph():
    G = eg.Graph()
    G.add_nodes_from(range(5))
    edgelist = np.empty((0, 2), dtype=int)
    node_attr = np.array([0, 1, 0, 1, 1])
    assortM, assortT, Z = localAssort(edgelist, node_attr)
    assert assortM.shape == (5, 10)
    assert np.all(np.isnan(assortM)) or np.allclose(assortM, 0, atol=1e-5)
    assert np.all(np.isnan(assortT)) or np.allclose(assortT, 0, atol=1e-5)
    assert np.all(np.isnan(Z)) or np.allclose(Z, 0, atol=1e-5)


def test_localassort_high_restart_probabilities():
    G = eg.path_graph(5)
    edgelist = np.array(list(G.edges))
    node_attr = np.array([1, 0, 1, 0, 1])
    pr = np.array([0.95, 0.99])
    assortM, assortT, Z = localAssort(edgelist, node_attr, pr=pr)
    assert assortM.shape == (5, 2)
    assert assortT.shape == (5,)
    assert Z.shape == (5,)


def test_localassort_invalid_attribute_length():
    edgelist = np.array([[0, 1], [1, 2]])
    node_attr = np.array([0, 1])  # too short
    with pytest.raises(ValueError):
        localAssort(edgelist, node_attr)
