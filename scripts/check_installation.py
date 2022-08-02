#!/usr/bin/env python3


import easygraph as eg


G = eg.Graph()

G.add_edge(1, 2)
assert G.edges == [(1, 2, {})]

G.add_node("hello world")
G.add_node("Jack", node_attr={"age": 10, "gender": "M"})
assert G.nodes == {
    1: {},
    2: {},
    "hello world": {},
    "Jack": {"node_attr": {"age": 10, "gender": "M"}},
}
