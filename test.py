import easygraph as eg
import hypernetx as hnx


data = {0: ("A", "B"), 1: ("B", "C"), 2: ("D", "A", "E"), 3: ("F", "G", "H", "D")}
# H = hnx.Hypergraph(data)
# print(list(H.nodes))
# print(list(H.edges))
# print('distance:',H.distance('A','E'))
# print(eg.__file__)
a = eg.Hypergraph(
    3,
    e_list=[[1, 2], [0, 1, 2]],
    v_property=[{"m": 1}, {"m": 2}, {"m": 3}],
    e_property=[{"name": "1"}, {"name": "2"}],
)
# a.add_hyperedges([[1,2,3],[2,4],[0,1],[0,1,2]])
# print(a.distance(1,4))
# print(a.adjacency_matrix(weight=True))
# print(a.get_linegraph(edge=False,weight=True).edges)
# print(a.get_linegraph(edge=True,weight=True).edges)
# print(a.incidence_matrix)
# print(a.diameter())
print(a.state_dict)
print(a.e)


# star_expansion
print(a.get_star_expeansion())
# eg.load_hypergraph_data(dataset=None)
b = eg.load_hypergraph_data(dataset="email-enron")
print("b", b)
print("b:", b.num_v, b.num_e)
print(2 in b.e[1])
# print("b.e[1]:",b.e[1])
# print("b_e:",b.e[0])
# print("b_raw",b.state_dict)
# print(len(b.group_names))
