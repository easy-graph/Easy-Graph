"""有向图原始形式正确性对比测试：对同一数据集分别计算 eg / ig / gt 的逐节点聚类，并输出差异。

运行示例:
  python test_correctness.py        # 打印表格（前100节点）与差异摘要

说明:
  - 使用 ArxivHEPTHDataset，保持原始有向图形式，不进行任何数据加工。
  - 依赖: easygraph, igraph, graph_tool
"""
import sys
from statistics import mean

import easygraph as eg
from easygraph.datasets import ArxivHEPTHDataset
import numpy as np

try:
    import igraph
except ImportError:
    igraph = None

try:
    import graph_tool.all as gt
except ImportError:
    gt = None


def main():
    data = ArxivHEPTHDataset()
    G_py = data[0]
    # 读取原始有向边，不做任何加工处理
    edges = [(u, v) for u, v, *_ in G_py.edges]

    # 按 easygraph 的节点顺序固定 nodes_order，保证三方对齐
    nodes_order = list(G_py.nodes)

    # easygraph: 使用原始有向边构造有向图，调用有向图聚类
    G_eg = eg.DiGraph()
    G_eg.add_edges_from(edges)
    eg_clust = eg.clustering(G_eg)  # 有向图聚类

    # igraph: 用 nodes_order 构造索引有向图，调用有向图聚类方法
    if igraph is None:
        print("igraph 未安装，跳过 igraph 比较")
        ig_clust = {}
    else:
        label2idx = {lab: i for i, lab in enumerate(nodes_order)}
        edges_idx = [(label2idx[u], label2idx[v]) for u, v in edges]
        G_ig = igraph.Graph(n=len(nodes_order), edges=edges_idx, directed=True)
        # 对于有向图，使用 transitivity_local_undirected 计算局部聚类
        # igraph 的有向图聚类需要特殊处理，使用 mode="zero" 处理度数 < 2 的节点
        vals = igraph.transitivity_local_undirected(G_ig, mode="zero")
        ig_clust = {nodes_order[i]: float(vals[i]) for i in range(len(vals))}

    # graph_tool: 创建所有顶点按 nodes_order 顺序，保证映射一致，使用有向图聚类
    if gt is None:
        print("graph_tool 未安装，跳过 graph_tool 比较")
        gt_clust = {}
    else:
        G_gt = gt.Graph(directed=True)
        vmap = {lab: G_gt.add_vertex() for lab in nodes_order}
        for u, v in edges:
            G_gt.add_edge(vmap[u], vmap[v])
        prop = G_gt.new_vertex_property("double")
        # local_clustering 在有向图上自动计算有向聚类
        gt.local_clustering(G_gt, prop=prop)
        gt_clust = {lab: float(prop[vmap[lab]]) for lab in nodes_order}

    # 以 nodes_order 作为比较基准
    nodes = nodes_order

    # Prepare rows: node, eg, ig, gt, |eg-ig|, |eg-gt|
    rows = []
    diffs_eg_ig = []
    diffs_eg_gt = []
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

    # Print table (first 100 rows)
    header = ("node", "eg", "ig", "gt", "|eg-ig|", "|eg-gt|")
    print("\n" + "  ".join(f"{h:>10}" for h in header))
    for r in rows[:100]:
        n, e, i, g, d_ig, d_gt = r
        print(
            f"{n:10d}  {e:10.6f}  {('' if i is None else f'{i:10.6f}')}  {('' if g is None else f'{g:10.6f}')}  {('' if d_ig is None else f'{d_ig:10.6f}')}  {('' if d_gt is None else f'{d_gt:10.6f}')}")

    # Summary
    def stats(lst):
        if not lst:
            return (None, None, None)
        return (mean(lst), max(lst), len(lst))

    m_ig, max_ig, count_ig = stats(diffs_eg_ig)
    m_gt, max_gt, count_gt = stats(diffs_eg_gt)

    # 严格验证：使用 numpy.allclose 检查按 nodes_order 对齐的值
    print("\nSummary:")
    print(f"  Baseline nodes (easygraph): {len(nodes)}")
    if igraph is not None:
        eg_vals = np.array([eg_clust.get(n, 0.0) for n in nodes], dtype=float)
        ig_vals = np.array([ig_clust.get(n, np.nan) for n in nodes], dtype=float)
        mask = ~np.isnan(ig_vals)
        allclose = np.allclose(eg_vals[mask], ig_vals[mask], atol=1e-12, equal_nan=True)
        mismatches = [
            (n, float(eg_clust.get(n, 0.0)), float(ig_clust.get(n, 0.0)))
            for n in nodes
            if mask[nodes.index(n)] and not np.isclose(eg_clust.get(n, 0.0), ig_clust.get(n, 0.0), atol=1e-12)
        ]
        print(f"  eg vs ig: allclose={allclose}  mean_abs={m_ig:.6f}  max_abs={max_ig:.6f}  compared={count_ig}")
        if mismatches:
            print("  示例不一致节点（最多20）:")
            for n, e, i in mismatches[:20]:
                print(f"    {n}: eg={e:.12f}  ig={i:.12f}  diff={abs(e-i):.12e}")
    if gt is not None:
        eg_vals = np.array([eg_clust.get(n, 0.0) for n in nodes], dtype=float)
        gt_vals = np.array([gt_clust.get(n, np.nan) for n in nodes], dtype=float)
        mask = ~np.isnan(gt_vals)
        allclose = np.allclose(eg_vals[mask], gt_vals[mask], atol=1e-12, equal_nan=True)
        mismatches = [
            (n, float(eg_clust.get(n, 0.0)), float(gt_clust.get(n, 0.0)))
            for n in nodes
            if mask[nodes.index(n)] and not np.isclose(eg_clust.get(n, 0.0), gt_clust.get(n, 0.0), atol=1e-12)
        ]
        print(f"  eg vs gt: allclose={allclose}  mean_abs={m_gt:.6f}  max_abs={max_gt:.6f}  compared={count_gt}")
        if mismatches:
            print("  示例不一致节点（最多20）:")
            for n, e, g in mismatches[:20]:
                print(f"    {n}: eg={e:.12f}  gt={g:.12f}  diff={abs(e-g):.12e}")


if __name__ == "__main__":
    main()
