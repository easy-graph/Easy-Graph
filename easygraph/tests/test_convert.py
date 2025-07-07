import sys

import pytest


np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
sp = pytest.importorskip("scipy")

import easygraph as eg

from easygraph.utils.misc import *


class TestConvertNumpyArray:
    def setup_method(self):
        self.G1 = eg.complete_graph(5)

    def assert_equal(self, G1, G2):
        assert nodes_equal(G1.nodes, G2.nodes)
        assert edges_equal(G1.edges, G2.edges, need_data=False)

    def identity_conversion(self, G, A, create_using):
        assert A.sum() > 0
        GG = eg.from_numpy_array(A, create_using=create_using)
        self.assert_equal(G, GG)
        GW = eg.to_easygraph_graph(A, create_using=create_using)
        self.assert_equal(G, GW)

    def test_identity_graph_array(self):
        "Conversion from graph to array to graph."
        A = eg.to_numpy_array(self.G1)
        self.identity_conversion(self.G1, A, eg.Graph())


class TestConvertPandas:
    def setup_method(self):
        self.rng = np.random.RandomState(seed=5)
        ints = self.rng.randint(1, 11, size=(3, 2))
        a = ["A", "B", "C"]
        b = ["D", "A", "E"]
        df = pd.DataFrame(ints, columns=["weight", "cost"])
        df[0] = a  # Column label 0 (int)
        df["b"] = b  # Column label 'b' (str)
        self.df = df

        mdf = pd.DataFrame([[4, 16, "A", "D"]], columns=["weight", "cost", 0, "b"])
        # self.mdf = df.append(mdf)
        self.mdf = pd.concat([df, mdf])

    def assert_equal(self, G1, G2):
        assert nodes_equal(G1.nodes, G2.nodes)
        assert edges_equal(G1.edges, G2.edges, need_data=False)

    def test_from_edgelist_multi_attr(self):
        Gtrue = eg.Graph(
            [
                ("E", "C", {"cost": 9, "weight": 10}),
                ("B", "A", {"cost": 1, "weight": 7}),
                ("A", "D", {"cost": 7, "weight": 4}),
            ]
        )
        G = eg.from_pandas_edgelist(self.df, 0, "b", ["weight", "cost"])
        self.assert_equal(G, Gtrue)

    def test_from_adjacency(self):
        Gtrue = eg.DiGraph(
            [
                ("A", "B"),
                ("B", "C"),
            ]
        )
        data = {
            "A": {"A": 0, "B": 0, "C": 0},
            "B": {"A": 1, "B": 0, "C": 0},
            "C": {"A": 0, "B": 1, "C": 0},
        }
        dftrue = pd.DataFrame(data, dtype=np.intp)
        df = dftrue[["A", "C", "B"]]
        G = eg.from_pandas_adjacency(df, create_using=eg.DiGraph())
        self.assert_equal(G, Gtrue)


class TestConvertScipy:
    def setup_method(self):
        self.G1 = eg.complete_graph(3)

    def assert_equal(self, G1, G2):
        assert nodes_equal(G1.nodes, G2.nodes)
        assert edges_equal(G1.edges, G2.edges, need_data=False)

    # skip if on python < 3.8
    @pytest.mark.skipif(
        sys.version_info < (3, 8), reason="requires python3.8 or higher"
    )
    def test_from_scipy(self):
        data = sp.sparse.csr_matrix([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        G = eg.from_scipy_sparse_matrix(data)
        self.assert_equal(self.G1, G)


def test_from_edgelist():
    edgelist = [(0, 1), (1, 2)]
    G = eg.from_edgelist(edgelist)
    assert sorted((u, v) for u, v, _ in G.edges) == [(0, 1), (1, 2)]


def test_from_dict_of_lists():
    d = {0: [1], 1: [2]}
    G = eg.to_easygraph_graph(d)
    assert sorted((u, v) for u, v, _ in G.edges) == [(0, 1), (1, 2)]


def test_from_dict_of_dicts():
    d = {0: {1: {}}, 1: {2: {}}}
    G = eg.to_easygraph_graph(d)
    assert sorted((u, v) for u, v, _ in G.edges) == [(0, 1), (1, 2)]


def test_from_numpy_array():
    G = eg.complete_graph(3)
    A = eg.to_numpy_array(G)
    G2 = eg.from_numpy_array(A)
    assert sorted((u, v) for u, v, _ in G.edges) == sorted(
        (u, v) for u, v, _ in G2.edges
    )


def test_from_pandas_edgelist():
    df = pd.DataFrame({"source": [0, 1], "target": [1, 2], "weight": [0.5, 0.7]})
    G = eg.from_pandas_edgelist(df, source="source", target="target", edge_attr=True)
    assert sorted((u, v) for u, v, _ in G.edges) == [(0, 1), (1, 2)]


def test_from_pandas_adjacency():
    df = pd.DataFrame([[0, 1], [1, 0]], columns=["A", "B"], index=["A", "B"])
    G = eg.from_pandas_adjacency(df)
    assert sorted((u, v) for u, v, _ in G.edges) == [("A", "B")]


def test_from_scipy_sparse_matrix():
    mat = sp.sparse.csr_matrix([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    G = eg.from_scipy_sparse_matrix(mat)
    expected_edges = [(0, 1), (1, 2)]
    assert sorted((u, v) for u, v, _ in G.edges) == expected_edges


def test_invalid_dict_type():
    class NotGraph:
        pass

    with pytest.raises(eg.EasyGraphError):
        eg.to_easygraph_graph(NotGraph())
