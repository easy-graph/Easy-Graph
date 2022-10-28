import unittest

import easygraph as eg


class WeakTieTest(unittest.TestCase):
    def test_weakTie(self):
        test_graph = eg.DiGraph(
            [
                (1, 6),
                (1, 5),
                (1, 2),
                (1, 3),
                (1, 7),
                (1, 8),
                (1, 9),
                (4, 3),
                (4, 5),
                (4, 1),
                (10, 9),
                (10, 100),
            ]
        )
        high_score_list, score_dict = eg.weakTie(test_graph, 0.2, 3)
        self.assertEqual(high_score_list, [6, 5, 2])
        self.assertEqual(
            score_dict,
            {
                1: 0,
                6: 0.07500000000000001,
                5: 0.07500000000000001,
                2: 0.07500000000000001,
                3: 0.07500000000000001,
                7: 0.07500000000000001,
                8: 0.07500000000000001,
                9: 0.07500000000000001,
                4: 0,
                10: 0,
                100: 0,
            },
        )


if __name__ == "__main__":
    unittest.main()
