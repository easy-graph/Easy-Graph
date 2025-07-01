import easygraph as eg
import pytest

from easygraph.utils.exception import EasyGraphError


class TestRingLatticeHypergraph:
    def test_valid_ring_lattice(self):
        H = eg.ring_lattice(n=10, d=3, k=4, l=1)
        assert isinstance(H, eg.Hypergraph)
        assert H.num_v == 10
        assert all(len(edge) == 3 for edge in H.e[0])

    def test_k_less_than_zero_raises_error(self):
        with pytest.raises(EasyGraphError, match="Invalid k value!"):
            eg.ring_lattice(n=10, d=3, k=-2, l=1)

    def test_k_less_than_two_warns(self):
        with pytest.warns(UserWarning, match="disconnected"):
            H = eg.ring_lattice(n=10, d=3, k=1, l=1)
            assert isinstance(H, eg.Hypergraph)

    def test_k_odd_warns(self):
        with pytest.warns(UserWarning, match="divisible by 2"):
            H = eg.ring_lattice(n=10, d=3, k=3, l=1)
            assert isinstance(H, eg.Hypergraph)

    def test_ring_lattice_with_d_eq_1(self):
        H = eg.ring_lattice(n=5, d=1, k=2, l=0)
        assert all(len(edge) == 1 for edge in H.e[0])

    def test_ring_lattice_with_overlap_zero(self):
        H = eg.ring_lattice(n=6, d=2, k=2, l=0)
        assert all(len(edge) == 2 for edge in H.e[0])

    def test_large_n(self):
        H = eg.ring_lattice(n=100, d=4, k=6, l=2)
        assert H.num_v == 100
        assert all(len(e) == 4 for e in H.e[0])

    def test_n_equals_1(self):
        H = eg.ring_lattice(n=1, d=1, k=2, l=0)
        assert H.num_v == 1
        assert isinstance(H, eg.Hypergraph)

    def test_k_zero(self):
        H = eg.ring_lattice(n=5, d=2, k=0, l=1)
        assert H.num_v == 5
        assert len(H.e[0]) == 0
