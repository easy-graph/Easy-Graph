import unittest

import easygraph as eg


class NOBETest(unittest.TestCase):

    def test_NOBE_GA_SH(self):
        test_two_graph = eg.Graph([(1, 2), (1, 3), (1, 4), (1, 5), (1, 6)])
        self.assertEqual(
            eg.NOBE_GA_SH(test_two_graph, K=2, topk=3),
            [2, 3, 4]
        )


if __name__ == '__main__':
    unittest.main()
