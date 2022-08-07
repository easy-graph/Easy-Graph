import unittest

import easygraph as eg


class ICCTest(unittest.TestCase):
    def test_ICC(self):
        test_graph = eg.Graph([(1, 2), (2, 3), (3, 10), (15, 25)])

        self.assertEqual(eg.ICC(test_graph, 5), [2, 3, 10, 15, 25])

        self.assertEqual(eg.ICC(test_graph, 3), [3, 15, 25])

    def test_BICC(self):
        test_graph = eg.Graph([(1, 2), (2, 3), (3, 10), (15, 25)])
        test_hole_spanners = 4
        test_n_candidates = 4
        test_level_n_nodes = 4

        self.assertEqual(
            eg.BICC(
                test_graph, test_hole_spanners, test_n_candidates, test_level_n_nodes
            ),
            [1, 2, 3, 10],
        )

        test_hole_spanners = 3
        test_n_candidates = 5
        test_level_n_nodes = 6

        self.assertEqual(
            eg.BICC(
                test_graph, test_hole_spanners, test_n_candidates, test_level_n_nodes
            ),
            [2, 3, 15],
        )

    def test_AP_BICC(self):
        test_graph = eg.Graph([(1, 2), (15, 25), (50, 100)])
        test_hole_spanners = 3
        test_n_candidates = 2
        test_level_n_nodes = 1

        self.assertEqual(
            eg.AP_BICC(
                test_graph, test_hole_spanners, test_n_candidates, test_level_n_nodes
            ),
            [1, 2, 15],
        )


if __name__ == "__main__":
    unittest.main()
