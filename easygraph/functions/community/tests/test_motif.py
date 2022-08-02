import easygraph as eg


class TestMotif:
    @classmethod
    def setup_class(self):
        self.G = eg.Graph()
        self.G.add_nodes_from([1, 2, 3, 4, 5])
        self.G.add_edges_from([(1, 3), (2, 3), (3, 4), (4, 5), (3, 5)])

    def test_esu(self):
        res = eg.enumerate_subgraph(self.G, 3)
        res = [list(x) for x in res]
        exp_res = [{1, 3, 4}, {1, 2, 3}, {1, 3, 5}, {2, 3, 5}, {2, 3, 4}, {3, 4, 5}]
        exp_res = [list(x) for x in exp_res]
        assert sorted(res) == sorted(exp_res)
