import easygraph as eg
import pytest

from easygraph.utils.exception import EasyGraphError


class TestClassic:
    def test_complete_hypergraph(self):
        print(eg.complete_hypergraph(10))
        assert eg.complete_hypergraph(10) is not None

    def test_random_hypergraph(self):
        import random

        import numpy as np

        n = 100
        k1 = {i: random.randint(1, 100) for i in range(n)}
        k2 = {i: sorted(k1.values())[i] for i in range(n)}
        H = eg.chung_lu_hypergraph(k1, k2)
        H2 = eg.watts_strogatz_hypergraph(n=n, d=10, k=16, p=0.5, l=3)
        k1 = {i: random.randint(1, n) for i in range(n)}
        k2 = {i: sorted(k1.values())[i] for i in range(n)}
        g1 = {i: random.choice([0, 1]) for i in range(n)}
        g2 = {i: random.choice([0, 1]) for i in range(n)}
        omega = np.array([[n // 2, 10], [10, n // 2]])
        H3 = eg.dcsbm_hypergraph(k1, k2, g1, g2, omega)

        assert H != None
        assert H2 != None
        assert H3 != None

    def test_simple_hypergraph(self):
        H = eg.star_clique(6, 7, 2)
        print(H)

    def test_uniform_hypergraph(self):
        n = 1000
        m = 3
        k = {0: 1, 1: 2, 2: 3, 3: 3}
        H = eg.uniform_hypergraph_configuration_model(k, m)
        print(H)

        H2 = eg.uniform_erdos_renyi_hypergraph(10, 5, 0.5, "prob")
        # H2 = eg.uniform_HSBM(n,5,[3,4,5],[0.5,0.5,0.5])
        print("H2:", H2)

        H3 = eg.uniform_HPPM(10, 6, 0.9, 10, 0.9)

        print("H3:", H3)


class TestHypergraphGenerators:
    def test_empty_hypergraph_default(self):
        hg = eg.empty_hypergraph()
        assert hg.num_v == 1
        assert len(hg.e[0]) == 0

    def test_empty_hypergraph_custom_size(self):
        hg = eg.empty_hypergraph(5)
        assert hg.num_v == 5
        assert len(hg.e[0]) == 0

    def test_complete_hypergraph_zero_nodes_raises(self):
        with pytest.raises(EasyGraphError):
            eg.complete_hypergraph(0)

    def test_complete_hypergraph_n_1_excludes_singletons(self):
        hg = eg.complete_hypergraph(1, include_singleton=False)
        assert hg.num_v == 1
        assert len(hg.e[0]) == 0

    def test_complete_hypergraph_n_3_excludes_singletons(self):
        hg = eg.complete_hypergraph(3, include_singleton=False)
        expected_edges = [[0, 1], [0, 2], [1, 2], [0, 1, 2]]
        assert sorted(sorted(e) for e in hg.e[0]) == sorted(
            sorted(e) for e in expected_edges
        )

    def test_complete_hypergraph_n_3_includes_singletons(self):
        hg = eg.complete_hypergraph(3, include_singleton=True)
        expected_edges = [[0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]]
        assert sorted(sorted(e) for e in hg.e[0]) == sorted(
            sorted(e) for e in expected_edges
        )
