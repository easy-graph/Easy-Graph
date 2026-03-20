"""从 TXT 边表读取图并对 eg/ig/gt 逐节点聚类结果做对齐比较。

用法示例:
  python test_correctness_from_txt.py                        # 使用默认 soc-Epinions1.txt
  python test_correctness_from_txt.py path/to/edges.txt     # 指定边表文件

边表格式假定为每行两个整数: "u v"（可含注释或空行），节点标签按文件中出现顺序收集。
其余比较逻辑与 `test_correctness.py` 保持一致（nodes_order 对齐、numpy.allclose 检查、不一致示例）。
"""
import sys
from statistics import mean
import numpy as np

import easygraph as eg
try:
    import igraph
except ImportError:
    igraph = None

try:
    import graph_tool.all as gt
except ImportError:
    gt = None


def read_edge_list(path):
    edges = []
    nodes_seen = []
    node_set = set()
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                u = int(parts[0])
                v = int(parts[1])
            except ValueError:
                # 尝试原样使用标签（字符串）
                u = parts[0]
                v = parts[1]
            edges.append((u, v))
            for x in (u, v):
                if x not in node_set:
                    node_set.add(x)
                    nodes_seen.append(x)
    return edges, nodes_seen


def build_graph_tool_from_edges(edges, nodes_order):
    G = gt.Graph(directed=False)
    vmap = {lab: G.add_vertex() for lab in nodes_order}
    for u, v in edges:
        if u == v:
            continue
        G.add_edge(vmap[u], vmap[v])
    return G, vmap


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'cpp_easygraph/functions/basic/test/soc-Epinions1.txt'
    edges, nodes_order = read_edge_list(path)

    # 将边无向化（去重自环）并按出现顺序对齐
    edges_ud = []
    seen = set()
    for u, v in edges:
        if u == v:
            continue
        key = (min(u, v), max(u, v))
        if key in seen:
            continue
        seen.add(key)
        edges_ud.append(key)

    # 构造 easygraph 无向图作为基准
    G_py = eg.Graph()
    G_py.add_edges_from(edges_ud)
    eg_clust = eg.clustering(G_py)

    # igraph: 用 nodes_order 作为顶点顺序
    if igraph is None:
        print('igraph 未安装，跳过 igraph 比较')
        ig_clust = {}
    else:
        label2idx = {lab: i for i, lab in enumerate(nodes_order)}
        edges_idx = [(label2idx[u], label2idx[v]) for u, v in edges_ud]
        G_ig = igraph.Graph(n=len(nodes_order), edges=edges_idx, directed=False)
        vals = G_ig.transitivity_local_undirected(mode='zero')
        ig_clust = {nodes_order[i]: float(vals[i]) for i in range(len(vals))}

    # graph_tool
    if gt is None:
        print('graph_tool 未安装，跳过 graph_tool 比较')
        gt_clust = {}
    else:
        G_gt, vmap = build_graph_tool_from_edges(edges_ud, nodes_order)
        prop = G_gt.new_vertex_property('double')
        gt.local_clustering(G_gt, prop=prop)
        gt_clust = {lab: float(prop[vmap[lab]]) for lab in nodes_order}

    # 对齐比较（按 nodes_order）
    nodes = nodes_order
    diffs_eg_ig = []
    diffs_eg_gt = []
    rows = []
    for n in nodes:
        e = float(eg_clust.get(n, 0.0))
        i = float(ig_clust.get(n, 0.0)) if n in ig_clust else None
        g = float(gt_clust.get(n, 0.0)) if n in gt_clust else None
        d_ig = abs(e - i) if i is not None else None
        d_gt = abs(e - g) if g is not None else None
        if d_ig is not None:
            diffs_eg_ig.append(d_ig)
        if d_gt is not None:
            diffs_eg_gt.append(d_gt)
        rows.append((n, e, i, g, d_ig, d_gt))

    header = ('node', 'eg', 'ig', 'gt', '|eg-ig|', '|eg-gt|')
    print('\n' + '  '.join(f'{h:>10}' for h in header))
    for r in rows[:100]:
        n, e, i, g, d_ig, d_gt = r
        print(f"{str(n):>10}  {e:10.6f}  {('' if i is None else f'{i:10.6f}')}  {('' if g is None else f'{g:10.6f}')}  {('' if d_ig is None else f'{d_ig:10.6f}')}  {('' if d_gt is None else f'{d_gt:10.6f}')}")

    def stats(lst):
        if not lst:
            return (None, None, None)
        return (mean(lst), max(lst), len(lst))

    m_ig, max_ig, count_ig = stats(diffs_eg_ig)
    m_gt, max_gt, count_gt = stats(diffs_eg_gt)

    print('\nSummary:')
    print(f'  Baseline nodes (easygraph): {len(nodes)}')

    if igraph is not None:
        eg_vals = np.array([eg_clust.get(n, 0.0) for n in nodes], dtype=float)
        ig_vals = np.array([ig_clust.get(n, np.nan) for n in nodes], dtype=float)
        mask = ~np.isnan(ig_vals)
        allclose = np.allclose(eg_vals[mask], ig_vals[mask], atol=1e-12, equal_nan=True)
        print(f'  eg vs ig: allclose={allclose}  mean_abs={m_ig:.6f}  max_abs={max_ig:.6f}  compared={count_ig}')

    if gt is not None:
        eg_vals = np.array([eg_clust.get(n, 0.0) for n in nodes], dtype=float)
        gt_vals = np.array([gt_clust.get(n, np.nan) for n in nodes], dtype=float)
        mask = ~np.isnan(gt_vals)
        allclose = np.allclose(eg_vals[mask], gt_vals[mask], atol=1e-12, equal_nan=True)
        print(f'  eg vs gt: allclose={allclose}  mean_abs={m_gt:.6f}  max_abs={max_gt:.6f}  compared={count_gt}')


if __name__ == '__main__':
    main()
