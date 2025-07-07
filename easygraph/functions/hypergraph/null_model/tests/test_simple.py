from itertools import combinations

import easygraph as eg
import pytest


class TestStarCliqueHypergraph:
    def test_valid_star_clique(self):
        H = eg.star_clique(n_star=5, n_clique=4, d_max=2)
        assert isinstance(H, eg.Hypergraph)
        assert H.num_v == 9  # 5 star nodes + 4 clique nodes
        assert any(0 in edge for edge in H.e[0])  # star center connected

    def test_minimum_valid_values(self):
        H = eg.star_clique(n_star=2, n_clique=2, d_max=1)
        assert H.num_v == 4
        assert len(H.e[0]) >= 2

    def test_n_star_zero_raises(self):
        with pytest.raises(ValueError, match="n_star must be an integer > 0."):
            eg.star_clique(0, 3, 1)

    def test_n_clique_zero_raises(self):
        with pytest.raises(ValueError, match="n_clique must be an integer > 0."):
            eg.star_clique(3, 0, 1)

    def test_d_max_negative_raises(self):
        with pytest.raises(ValueError, match="d_max must be an integer >= 0."):
            eg.star_clique(3, 4, -1)

    def test_d_max_too_large_raises(self):
        with pytest.raises(ValueError, match="d_max must be <= n_clique - 1."):
            eg.star_clique(3, 4, 5)

    def test_no_clique_edges_if_d_max_zero(self):
        H = eg.star_clique(3, 3, 0)
        clique_nodes = set(range(3, 6))
        for edge in H.e[0]:
            assert not clique_nodes.issubset(edge)

    def test_clique_hyperedges_match_combinations(self):
        n_star, n_clique, d_max = 3, 4, 2
        H = eg.star_clique(n_star, n_clique, d_max)
        clique_nodes = list(range(n_star, n_star + n_clique))
        expected = {
            tuple(sorted(e))
            for d in range(1, d_max + 1)
            for e in combinations(clique_nodes, d + 1)
        }
        actual = {
            tuple(sorted(e)) for e in H.e[0] if all(node in clique_nodes for node in e)
        }
        assert expected.issubset(actual)

    def test_star_legs_connect_to_center(self):
        H = eg.star_clique(5, 4, 1)
        star_nodes = list(range(5))
        center = star_nodes[0]
        for i in range(1, 4):  # last star leg is used to connect to clique
            assert any({center, i}.issubset(edge) for edge in H.e[0])
