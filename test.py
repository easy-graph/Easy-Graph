import easygraph as eg
a = eg.Graph()
b = eg.GraphC()
a.add_edge(0,1)
b.add_edge(0,1)
print(eg.k_core(a))
print(eg.k_core(b))