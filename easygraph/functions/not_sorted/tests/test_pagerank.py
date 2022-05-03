import io
import time
import pytest
import sys

import easygraph as eg
np = pytest.importorskip("numpy")

class TestPageRank:
    def setup_method(self):
        self.G1 = eg.complete_graph(5)
        
    def test_pagerank(self):
        pg_true = {
            0: 0.21366099794430937,
            1: 0.15107456550177534,
            2: 0.21080171930480293,
            3: 0.2136609979443094,
            4: 0.21080171930480288
        }
        self.G1.remove_edge(0,3)
        self.G1.remove_edge(1,2)
        self.G1.remove_edge(1,4)
        pg = eg.pagerank(self.G1)
        for k, v in pg.items():
            assert pytest.approx(v, 0.0000001) == pg_true[k]