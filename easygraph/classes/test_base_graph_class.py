import os
import sys
import time

import easygraph as eg


print(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "../cpp_easygraph")
    )
)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import easygraph.classes as cls  # Spend 4.9s on importing this damn big lib.


def test_iter():
    g = eg.Graph()
    # repeated endings test
    g.add_edge(None, None)  # 1
    g.add_edge(True, False)

    g.add_edge(0b1000, 100)
    print(g.edges)


test_iter()
