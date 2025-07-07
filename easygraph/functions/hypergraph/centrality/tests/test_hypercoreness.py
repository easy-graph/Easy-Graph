import unittest

import easygraph as eg


class TestHypercoreness(unittest.TestCase):
    def test_simple_hypergraph(self):
        hg = eg.Hypergraph(num_v=5, e_list=[(0, 1), (1, 2, 3), (3, 4)])
        si = eg.size_independent_hypercoreness(hg)
        fb = eg.frequency_based_hypercoreness(hg)

        self.assertIsInstance(si, dict)
        self.assertIsInstance(fb, dict)
        self.assertTrue(set(si.keys()).issubset(set(hg.v)))
        self.assertTrue(set(fb.keys()).issubset(set(hg.v)))

        for val in si.values():
            self.assertIsInstance(val, float)
            self.assertGreaterEqual(val, 0)

        for val in fb.values():
            self.assertIsInstance(val, float)
            self.assertGreaterEqual(val, 0)

    def test_single_hyperedge(self):
        hg = eg.Hypergraph(num_v=3, e_list=[(0, 1, 2)])
        si = eg.size_independent_hypercoreness(hg)
        fb = eg.frequency_based_hypercoreness(hg)

        self.assertTrue(all(v >= 0 for v in si.values()))
        self.assertTrue(all(v >= 0 for v in fb.values()))

    def test_large_uniform_hypergraph(self):
        hg = eg.Hypergraph(num_v=10, e_list=[(i, i + 1, i + 2) for i in range(7)])
        si = eg.size_independent_hypercoreness(hg)
        fb = eg.frequency_based_hypercoreness(hg)

        self.assertEqual(len(si), 10)
        self.assertEqual(len(fb), 10)

    def test_empty_hypergraph_raises(self):
        hg = eg.Hypergraph(num_v=1, e_list=[])
        with self.assertRaises(IndexError):
            eg.size_independent_hypercoreness(hg)

        with self.assertRaises(IndexError):
            eg.frequency_based_hypercoreness(hg)


if __name__ == "__main__":
    unittest.main()
