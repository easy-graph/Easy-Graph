import unittest

import easygraph as eg
import numpy as np


class ConvertToMatrix(unittest.TestCase):
    def test_to_numpy_matrix(self):
        test_graph = eg.empty_graph(2)
        numpy_matrix = eg.to_numpy_matrix(test_graph).tolist()
        length_of_test_graph = len(test_graph)
        test_array = np.full((length_of_test_graph, length_of_test_graph), 0.0).tolist()
        self.assertEqual(numpy_matrix, test_array)  # add assertion here

    def test_from_numpy_array(self):
        test_array = np.array([[1, 1], [2, 1]])
        test_graph = eg.from_numpy_array(test_array)
        test_multigraph = eg.from_numpy_array(test_array, create_using=eg.MultiGraph)
        parallel_edges_graph = eg.from_numpy_array(
            test_array, parallel_edges=True, create_using=eg.MultiGraph
        )
        expected_return = [
            tuple((0, 0, {"weight": 1})),
            tuple((0, 1, {"weight": 2})),
            tuple((1, 1, {"weight": 1})),
        ]

        multigraph_and_parallel_edges_expected_return = [
            tuple((0, 0, 0, {"weight": 1})),
            tuple((0, 1, 0, {"weight": 1})),
            tuple((1, 1, 0, {"weight": 1})),
        ]

        data_types = [("weight", float), ("cost", int)]
        array_with_data_types = np.array([[(1.0, 2)]], dtype=data_types)
        graph_with_data_types = eg.from_numpy_array(array_with_data_types)
        array_with_data_types_expected_return = [
            tuple((0, 0, {"weight": 1.0, "cost": 2}))
        ]
        self.assertListEqual(test_graph.edges, expected_return)
        self.assertEqual(
            test_multigraph.edges, multigraph_and_parallel_edges_expected_return
        )
        self.assertEqual(
            parallel_edges_graph.edges, multigraph_and_parallel_edges_expected_return
        )
        self.assertEqual(
            graph_with_data_types.edges, array_with_data_types_expected_return
        )

        # Wrong dimensions
        array_with_wrong_dimensions = np.array([[1, 1], [2, 1]], ndmin=3)
        with self.assertRaises(eg.EasyGraphError):
            eg.from_numpy_array(array_with_wrong_dimensions)

    def test_to_numpy_array(self):
        test_graph = eg.Graph([(1, 2)])
        test_array = eg.to_numpy_array(test_graph)
        expected_return = np.array([[0.0, 1.0], [1.0, 0.0]])
        self.assertEqual(test_array.tolist(), expected_return.tolist())

        test_multi_di_graph = eg.MultiDiGraph()
        test_multi_di_graph.add_edge(0, 1, weight=2)
        test_multi_di_graph.add_edge(1, 0)
        test_multi_di_graph.add_edge(2, 2, weight=3)
        test_multi_di_graph.add_edge(2, 2)
        multi_di_graph_test_array = eg.to_numpy_array(
            test_multi_di_graph, nodelist=[0, 1, 2]
        ).tolist()
        multi_di_graph_expected_return = np.array(
            [[0.0, 2.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 4.0]]
        ).tolist()
        self.assertEqual(multi_di_graph_test_array, multi_di_graph_expected_return)

    def test_from_pandas_adjacency(self):
        import pandas as pd

        test_dataframe = pd.DataFrame([[1, 1, 3], [2, 1]])
        test_graph = eg.from_pandas_adjacency(test_dataframe)

        test_nodes_expected_return = {0: {}, 1: {}}
        test_edges_expected_return = [
            (0, 0, {"weight": 1}),
            (0, 1, {"weight": 2}),
            (1, 1, {"weight": 1}),
        ]

        self.assertEqual(test_graph.nodes, test_nodes_expected_return)
        self.assertEqual(test_graph.edges, test_edges_expected_return)

    def test_from_pandas_edgelist(self):
        import pandas as pd

        ints = np.array([[4, 7], [7, 1], [10, 9]])
        a = ["x1", "x2", "x3"]
        b = ["y1", "y2", "y3"]
        df = pd.DataFrame(ints, columns=["test_col_one", "test_col_two"])
        df[0] = a
        df["b"] = b
        pandas_edge_list_graph = eg.from_pandas_edgelist(
            df, 0, "b", ["test_col_one", "test_col_two"]
        )
        pandas_edge_list_graph_edges_expected_return = [
            ("x1", "y1", {"test_col_one": 4, "test_col_two": 7}),
            ("x2", "y2", {"test_col_one": 7, "test_col_two": 1}),
            ("x3", "y3", {"test_col_one": 10, "test_col_two": 9}),
        ]

        pandas_edge_list_graph_nodes_expected_return = {
            "x1": {},
            "y1": {},
            "x2": {},
            "y2": {},
            "x3": {},
            "y3": {},
        }

        pandas_edge_list_graph_adj_expected_return = {
            "x1": {"y1": {"test_col_one": 4, "test_col_two": 7}},
            "y1": {"x1": {"test_col_one": 4, "test_col_two": 7}},
            "x2": {"y2": {"test_col_one": 7, "test_col_two": 1}},
            "y2": {"x2": {"test_col_one": 7, "test_col_two": 1}},
            "x3": {"y3": {"test_col_one": 10, "test_col_two": 9}},
            "y3": {"x3": {"test_col_one": 10, "test_col_two": 9}},
        }

        self.assertEqual(
            pandas_edge_list_graph.edges, pandas_edge_list_graph_edges_expected_return
        )
        self.assertEqual(
            pandas_edge_list_graph.nodes, pandas_edge_list_graph_nodes_expected_return
        )
        self.assertEqual(
            pandas_edge_list_graph.adj, pandas_edge_list_graph_adj_expected_return
        )

    def test_from_scipy_sparse_matrix(self):
        import scipy as sp

        test_array = sp.sparse.eye(3, 3, 1)
        test_graph = eg.from_scipy_sparse_matrix(test_array)
        from_scipy_sparse_matrix_nodes_expected_return = {0: {}, 1: {}, 2: {}}
        from_scipy_sparse_matrix_edges_expected_return = [
            (0, 1, {"weight": 1.0}),
            (1, 2, {"weight": 1.0}),
        ]
        from_scipy_sparse_matrix_adj_expected_return = {
            0: {1: {"weight": 1.0}},
            1: {0: {"weight": 1.0}, 2: {"weight": 1.0}},
            2: {1: {"weight": 1.0}},
        }

        self.assertEqual(
            test_graph.nodes, from_scipy_sparse_matrix_nodes_expected_return
        )
        self.assertEqual(
            test_graph.edges, from_scipy_sparse_matrix_edges_expected_return
        )
        self.assertEqual(test_graph.adj, from_scipy_sparse_matrix_adj_expected_return)

    # skip on python<3.8
    @unittest.skipUnless(
        hasattr(np, "sparse") and hasattr(np.sparse, "csr_array"),
        "skip if np.sparse.csr_array is not available",
    )
    def test_from_scipy_sparse_array(self):
        import scipy as sp

        test_cols = np.array([1, 2, 2, 2, 0, 0])
        test_rows = np.array([0, 1, 2, 0, 2, 2])
        test_data = np.array([1, 2, 3, 4, 5, 6])
        test_array = sp.sparse.csr_array(
            (test_data, (test_rows, test_cols)), shape=(3, 3)
        )
        test_graph = eg.from_scipy_sparse_array(test_array)

        from_scipy_sparse_array_nodes_expected_return = {0: {}, 1: {}, 2: {}}

        from_scipy_sparse_array_edges_expected_return = [
            (0, 1, {"weight": 1}),
            (0, 2, {"weight": 11}),
            (1, 2, {"weight": 2}),
            (2, 2, {"weight": 3}),
        ]

        from_scipy_sparse_array_adj_expected_return = {
            0: {1: {"weight": 1}, 2: {"weight": 11}},
            1: {0: {"weight": 1}, 2: {"weight": 2}},
            2: {0: {"weight": 11}, 1: {"weight": 2}, 2: {"weight": 3}},
        }

        self.assertEqual(
            test_graph.nodes, from_scipy_sparse_array_nodes_expected_return
        )
        self.assertEqual(
            test_graph.edges, from_scipy_sparse_array_edges_expected_return
        )
        self.assertEqual(test_graph.adj, from_scipy_sparse_array_adj_expected_return)


if __name__ == "__main__":
    unittest.main()
