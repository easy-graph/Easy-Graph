import unittest

import easygraph as eg


class ModularityTest(unittest.TestCase):

    def test_modularity_graph(self):
        test_graph = eg.Graph([(1, 2)])
        modularity_expected_result = -0.5
        modularity_actual_result = self._run_modularity_with_test_graph(test_graph)
        self.assertEqual(modularity_expected_result, modularity_actual_result)

    def test_modularity_digraph(self):
        test_graph = eg.DiGraph(
            [
                (4, 6),
                (6, 8),
                (8, 10),
                (12, 14)
            ]
        )
        modularity_expected_result = -0.125
        modularity_actual_result = self._run_modularity_with_test_graph(test_graph)
        self.assertEqual(modularity_expected_result, modularity_actual_result)

    def _run_modularity_with_test_graph(self, test_graph):
        test_communities = {i: frozenset([i]) for i in range(len(test_graph))}
        test_labels_for_nodes = {key: val for key, val in enumerate(test_graph.nodes)}
        test_partition = [[test_labels_for_nodes[x] for x in community] for community in test_communities.values()]
        return eg.modularity(test_graph, test_partition)


if __name__ == '__main__':
    unittest.main()
