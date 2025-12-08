import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.datasets import Coauthor, Planetoid, Reddit, Flickr, Yelp, Amazon
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
import torch
import igraph as ig

# -----------------------------
# 指标计算函数
# -----------------------------
# def degree_heterogeneity(G):
#     degrees = np.array([d for _, d in G.degree()])
#     mean_k = degrees.mean()
#     mean_k2 = np.mean(degrees ** 2)
#     H = mean_k2 / (mean_k ** 2) if mean_k > 0 else 0
#     return H

# def num_com(G, min_size=10):
#     """
#     使用 igraph 统计 NetworkX 图的社区数Louvain / Multilevel。
#     G: networkx.Graph
#     返回: 社区数量
#     """
#     ig_g = ig.Graph.from_networkx(G)
#     communities = ig_g.community_multilevel()
#     # 过滤掉太小的社区
#     filtered_coms = [c for c in communities if len(c) >= min_size]
#     return len(filtered_coms)

# def load_snap_graph(dataset_name, root='/root/autodl-tmp/data'):
#     path_map = {
#         'com-lj': 'com-lj/soc-LiveJournal1.txt',
#         'com-amazon': 'com-amazon/com-amazon.ungraph.txt',
#         'pokec': 'pokec/soc-pokec-relationships.txt'
#     }
#     path = f"{root}/{path_map[dataset_name]}"
#     G = nx.Graph()
#     with open(path, 'r') as f:
#         for line in f:
#             if line.startswith('#'):  # 跳过注释
#                 continue
#             u, v = map(int, line.strip().split())
#             G.add_edge(u, v)
#     return G

# # -----------------------------
# # 需要处理的数据集
# # -----------------------------
# DATASETS = [
#     ('Coauthor', 'CS'),
#     ('Coauthor', 'Physics'),
#     ('Planetoid', 'Cora'),
#     ('Planetoid', 'Citeseer'),
#     ('Planetoid', 'PubMed'),
#     ('ogb', 'ogbn-arxiv'),
#     ('ogb', 'ogbn-products'),
#     ('Reddit', 'Reddit'),
#     ('Flickr', 'Flickr'),
#     ('Yelp', 'Yelp'),
#     ('snap', 'com-lj'),
#     ('snap', 'com-amazon'),
#     ('snap', 'pokec')
# ]

# # -----------------------------
# # 主程序
# # -----------------------------
# transform = T.NormalizeFeatures()

# _load = torch.load
# def load(*args, **kwargs):
#     kwargs['weights_only'] = False
#     return _load(*args, **kwargs)
# torch.load = load

# results = []
# for backend_type, dataset_name in DATASETS:
#     print(f"\n================= 数据集: {dataset_name} =================")

#     # -------------------- 加载数据集 --------------------
#     if backend_type == 'Coauthor':
#         dataset = Coauthor(root=f'/root/Easy-Graph/easygraph/nn/tests/data/Coauthor', name=dataset_name)
#         edge_index = dataset[0].edge_index.numpy()
#         G = nx.Graph()
#         G.add_edges_from(edge_index.T)

#     elif backend_type == 'Planetoid':
#         dataset = Planetoid(root=f'/tmp/{dataset_name}', name=dataset_name, transform=transform)
#         edge_index = dataset[0].edge_index.numpy()
#         G = nx.Graph()
#         G.add_edges_from(edge_index.T)

#     elif backend_type in ['Reddit', 'Flickr', 'Yelp']:
#         cls_map = {'Reddit': Reddit, 'Flickr': Flickr, 'Yelp': Yelp}
#         dataset = cls_map[backend_type](root=f'/root/autodl-tmp/data/{backend_type}')
#         edge_index = dataset[0].edge_index.numpy()
#         G = nx.Graph()
#         G.add_edges_from(edge_index.T)

#     elif backend_type == 'ogb':
#         dataset = PygNodePropPredDataset(name=dataset_name, root='/root/Easy-Graph/easygraph/nn/tests/data/OGB')
#         edge_index = dataset[0].edge_index.numpy()
#         G = nx.Graph()
#         G.add_edges_from(edge_index.T)

#     elif backend_type == 'snap':
#         # SNAP 格式的 txt 文件
#         G = load_snap_graph(dataset_name, root='/root/autodl-tmp/data')

#     else:
#         raise ValueError(f"Unknown dataset type {backend_type} or dataset {dataset_name}")

#     # -------------------- 计算指标 --------------------
#     print(f'dataset: {dataset_name} load success!')
#     H = degree_heterogeneity(G)
#     NC = num_com(G)
#     N = G.number_of_nodes()

#     results.append((dataset_name, H, NC, N))
#     print(f"{dataset_name}: H={H:.4f}, Num_Communities={NC}, N={N}")

#     N = G.number_of_nodes()       # 节点数
#     E = G.number_of_edges()       # 边数

#     # 新增的指标
#     avg_degree = 2 * E / N if N > 0 else 0        # 平均度
#     density = 2 * E / (N * (N - 1)) if N > 1 else 0  # 密度
#     edge_node_ratio = E / N if N > 0 else 0       # 边节点比

#     print(f"{dataset_name}: "
#         f"AvgDeg={avg_degree:.4f}, "
#         f"Density={density:.6f}, "
#         f"E/N={edge_node_ratio:.4f}"
#         f"E={E}")


# 度异质性，社区数量，节点数，平均度，密度，边节点比, 边数

# results = [
#     ("Cora", 2.7986, 104, 2708, 3.8981, 0.001440, 1.9490, 5278),
#     ("Citeseer", 2.4900, 420, 3279, 2.7765, 0.000847, 1.3882, 4552),
#     ("CS", 2.0390, 28, 18333, 8.9341, 0.000487, 4.4670, 81894),
#     ("PubMed", 3.7317, 47, 19717, 4.4960, 0.000228, 2.2480, 44324),
#     ("Physics", 2.1732, 22, 34493, 14.3775, 0.000417, 7.1888, 247962),
#     ("Flickr", 10.9179, 25, 89250, 10.0813, 0.000113, 5.0406,449878),
#     ("ogbn-arxiv", 26.1964, 143, 169343, 13.6740, 0.000081, 6.8370, 1157799),
#     ("Reddit", 3.6429, 26, 232965, 491.9876, 0.002112, 245.9938, 57307946),
#     ("com-amazon", 2.5221, 153, 403394, 12.1143, 0.000030, 6.0571, 2443408),
#     ("Yelp", 12.9901, 21759, 716847, 20.4669, 0.000029, 10.2335, 7335833),
#     ("pokec", 3.4607, 35, 1632803, 27.3174, 0.000017, 13.6587, 22301964),
#     ("ogbn-products", 4.5131, 4427, 2400608, 51.5362, 0.000021, 25.7681, 61859140),
#     ("com-lj", 9.4915, 5247, 4847571, 17.8933, 0.000004, 8.9467, 7335833)
# ]

# 度异质性，社区数量(过滤小社区后)，节点数，平均度，密度，边节点比, 边数
results = [
    ("Cora", 2.7986, 24, 2708, 3.8981, 0.001440, 1.9490, 5278),
    ("Citeseer", 2.4900, 39, 3279, 2.7765, 0.000847, 1.3882, 4552),
    ("CS", 2.0390, 23, 18333, 8.9341, 0.000487, 4.4670, 81894),
    ("PubMed", 3.7317, 43, 19717, 4.4960, 0.000228, 2.2480, 44324),
    ("Physics", 2.1732, 21, 34493, 14.3775, 0.000417, 7.1888, 247962),
    # ("Flickr", 10.9179, 26, 89250, 10.0813, 0.000113, 5.0406,449878),
    ("ogbn-arxiv", 26.1964, 103, 169343, 13.6740, 0.000081, 6.8370, 1157799),
    ("Reddit", 3.6429, 27, 232965, 491.9876, 0.002112, 245.9938, 57307946),
    # ("com-amazon", 2.5221, 164, 403394, 12.1143, 0.000030, 6.0571, 2443408),
    ("Yelp", 12.9901, 313, 716847, 20.4669, 0.000029, 10.2335, 7335833),
    # ("pokec", 3.4607, 33, 1632803, 27.3174, 0.000017, 13.6587, 22301964),
    ("ogbn-products", 4.5131, 204, 2400608, 51.5362, 0.000021, 25.7681, 61859140),
    ("com-lj", 9.4915, 1235, 4847571, 17.8933, 0.000004, 8.9467, 7335833)
]



plt.figure(figsize=(8,6))

for name, H, NC, N, AvgDeg, Density, ENR, E in results:
    plt.scatter(np.log1p(H), np.log10(NC), s=np.sqrt(N)/2, alpha=0.7, label=name)
    # plt.scatter(H, np.log10(NC), s=AvgDeg*10, alpha=0.7, label=name)  # 放大一点方便区分
    # plt.scatter(H, np.log10(NC), s= np.sqrt(E)/2 , alpha=0.7, label=name)

plt.xlabel("Degree Heterogeneity (H)")
plt.ylabel("Number of Communities")
plt.title("Real-world Network Analysis")
plt.grid(True, linestyle="--", alpha=0.6)

# 添加图例
plt.legend(fontsize=8, ncol=2, markerscale=0.5)

# 保存图片
plt.savefig("network_scatter_NC10.png", dpi=300, bbox_inches='tight')
plt.show()

