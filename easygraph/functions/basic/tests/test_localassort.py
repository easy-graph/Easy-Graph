import random

import easygraph as eg
import numpy as np

from matplotlib import pyplot as plt


class TestLocalAssort:
    @classmethod
    def setup_class(self):
        self.G = eg.get_graph_karateclub()
        random_value = [1, 2, 3, 4, 5]
        edgelist = []
        valuelist = []
        node_num = len(self.G.nodes)
        for e in self.G.edges:
            edgelist.append([e[0] - 1, e[1] - 1])

            # valuelist.append(random.choice(random_value))
        print("edgelist:", edgelist)
        for i in range(0, node_num):
            valuelist.append(random.choice(random_value))
        self.edgelist = np.int32(edgelist)
        valuelist = np.int32(valuelist)
        self.valuelist = valuelist

    def test_karateclub(self):
        assortM, assortT, Z = eg.localAssortF(
            self.edgelist, self.valuelist, pr=np.arange(0, 1, 0.1)
        )
        print("M:", assortM)
        print("T:", assortT)
        print("Z:", Z)

        _, assortT, Z = easygraph.functions.basic.localassort.localAssortF(
            self.edgelist, self.valuelist, pr=np.array([0.9])
        )
        print("_:", _)
        print("assortT:", assortT)
        print("Z:", Z)

        plt.hist(assortM)
        plt.show()
