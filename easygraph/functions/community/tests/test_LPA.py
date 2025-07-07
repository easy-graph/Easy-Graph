import unittest

import easygraph as eg


class TestLabelPropagationAlgorithms(unittest.TestCase):
    def setUp(self):
        self.graph_simple = eg.Graph()
        self.graph_simple.add_edges_from([(0, 1), (1, 2), (3, 4)])

        self.graph_weighted = eg.Graph()
        self.graph_weighted.add_edges_from(
            [
                (0, 1, {"weight": 3}),
                (1, 2, {"weight": 2}),
                (2, 0, {"weight": 4}),
                (3, 4, {"weight": 1}),
            ]
        )

        self.graph_disconnected = eg.Graph()
        self.graph_disconnected.add_edges_from([(0, 1), (2, 3), (4, 5)])

        self.graph_single_node = eg.Graph()
        self.graph_single_node.add_node(42)

        self.graph_empty = eg.Graph()

    def test_lpa(self):
        self.assertEqual(eg.functions.community.LPA(self.graph_single_node), {1: [42]})
        self.assertTrue(eg.functions.community.LPA(self.graph_simple))
        self.assertTrue(eg.functions.community.LPA(self.graph_weighted))
        self.assertTrue(eg.functions.community.LPA(self.graph_disconnected))

    def test_slpa(self):
        self.assertEqual(
            eg.functions.community.SLPA(self.graph_single_node, T=5, r=0.01), {1: [42]}
        )
        self.assertTrue(eg.functions.community.SLPA(self.graph_simple, T=10, r=0.1))
        self.assertTrue(
            eg.functions.community.SLPA(self.graph_disconnected, T=15, r=0.1)
        )

    def test_hanp(self):
        self.assertEqual(
            eg.functions.community.HANP(self.graph_single_node, m=0.1, delta=0.05),
            {1: [42]},
        )
        self.assertTrue(
            eg.functions.community.HANP(self.graph_simple, m=0.3, delta=0.1)
        )
        self.assertTrue(
            eg.functions.community.HANP(self.graph_weighted, m=0.5, delta=0.2)
        )

    def test_bmlpa(self):
        self.assertEqual(
            eg.functions.community.BMLPA(self.graph_single_node, p=0.1), {1: [42]}
        )
        self.assertTrue(eg.functions.community.BMLPA(self.graph_simple, p=0.3))
        self.assertTrue(eg.functions.community.BMLPA(self.graph_weighted, p=0.2))


if __name__ == "__main__":
    unittest.main()
