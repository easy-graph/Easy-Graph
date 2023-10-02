import easygraph as eg
import json
from easygraph.functions.community.louvain import *
G=eg.Graph()
G.add_edges([(2, 3), (1, 3), (3, 4), (4, 5),(6,7),(6,7),(3,7),(5,8),(1,6)])
d=eg.louvain_partitions(G)
for c in d:
   print(c)
