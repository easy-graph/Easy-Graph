import os
import sys
import time

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..', '..')))
import easygraph as eg  # Spend 4.9s on importing this damn big lib.


"""
def test_iter():
    g = eg.Graph()
    # tests of corner cases

    g.add_edge(0, 0)
    g.add_edge(True, False)
    g.add_edge(False, 1)
    g.add_edge(0b1000, 0x00a, edge_attr={"age": 19, "gender": "Male"})
    # g.add_edge(None, None)  # this shall result in an AssertionError
    # g.add_edge(None, 1)  # this shall result in an AssertionError
    # g.add_edge(1, None)  # this shall result in an AssertionError

    # g.add_edges(None)  # Triggers a TypeError saying that len() is not applicable to None
    g.add_edges([(True, False), ("Beijing National", "Day School")], [{}, {"Rating": 100}])
    g.add_node("FuDan Univ", node_attr={"faculty": 10000})  # 1.
    g.add_edge("Beijing National", "FuDan Univ")
    # g.add_node([]) # this shall result in an unhashable error
    g.add_node('Jack', node_attr={
        'age': 10,
        'gender': 'M'
    })
    # g.remove_node("Beijing National")
    g.remove_edges([('Day School', 'Beijing National')])
    # g.add_edges_from()

    print(g.add_extra_selfloop())


    g.nbr_v()
    g.nbunch_iter()
    g.from_hypergraph_hypergcn()
    # print(g._adj[8].get(10))
    print(g.edges)
    print(g.nodes)


test_iter()
"""

from easygraph.datasets import get_graph_karateclub


G = get_graph_karateclub()
# Calculate five shs(Structural Hole Spanners) in G
shs = eg.common_greedy(G, 5)
# Draw the Graph, and the shs is marked by a red star
eg.draw_SHS_center(G, shs)
# Draw CDF curves of "Number of Followers" of SH spanners and ordinary users in G.
eg.plot_Followers(G, shs)

import easygraph as eg


G = eg.Graph()
G.add_edge(1, 2)  # Add a single edge
print(G.edges)

G.add_edges([(2, 3), (1, 3), (3, 4), (4, 5), ((1, 2), (3, 4))])  # Add edges
print(G.edges)


G.add_node("hello world")
G.add_node("Jack", node_attr={"age": 10, "gender": "M"})
print(G.nodes)

# G.remove_nodes(['hello world','Tom','Lily','a','b'])#remove edges
G.remove_nodes(["hello world"])
print(G.nodes)

G.remove_edge(4, 5)
print(G.edges)

print(len(G))  # __len__(self)
for x in G:  # __iter__(self)
    print(x)
print(G[1])  # return list(self._adj[node].keys()) __contains__ __getitem__

for neighbor in G.neighbors(node=2):
    print(neighbor)

G.add_edges(
    [(1, 2), (2, 3), (1, 3), (3, 4), (4, 5)],
    edges_attr=[
        {"weight": 20},
        {"weight": 10},
        {"weight": 15},
        {"weight": 8},
        {"weight": 12},
    ],
)  # add weighted edges
G.add_node(6)
print(G.edges)

print(G.degree())
print(G.degree(weight="weight"))

G_index_graph, index_of_node, node_of_index = G.to_index_node_graph()
print(G_index_graph.adj)

G1 = G.copy()
print(G1.adj)

print(eg.effective_size(G))
